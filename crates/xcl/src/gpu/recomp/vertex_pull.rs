//! Experimental mesh-free raster pass for the existing NGG triangle records.
//!
//! No runtime dispatch uses this yet. The compute pass must finish writing res,
//! then the caller must synchronize it, bind a compatible graphics root signature,
//! and issue a triangle-list draw with the checked vertex count below.
//! Primitive-ID semantics and non-triangle topology are not handled here.

use super::{ranked, MsInterface, MESH_HEADER};
use crate::gpu::dxbc::SigElement;
use std::fmt::Write;

/// One non-indexed vertex for every corner of every possible triangle. Unused
/// slots are clipped by the VS. Limit both D3D12 draw count and byte offsets to u32.
pub fn draw_vertices(ms: &MsInterface, records: u32) -> Result<u32, &'static str> {
    if ms.max_vertices == 0 || ms.max_primitives == 0 {
        return Err("empty mesh record capacity");
    }
    let bytes = ms.record_dwords().checked_mul(4)
        .and_then(|n| n.checked_mul(records as usize))
        .ok_or("mesh record byte count overflow")?;
    if bytes > u32::MAX as usize {
        return Err("mesh records exceed 32-bit byte offsets");
    }
    ms.max_primitives.checked_mul(3)
        .and_then(|n| n.checked_mul(records))
        .ok_or("mesh fallback draw count overflow")
}

/// Generate a VS 6.6 that reads the same record as the mesh pass-through.
/// Draw with start vertex zero and a triangle list. Use no input layout.
/// Invalid primitive slots emit three identical positions outside the clip volume.
/// No vertex data is read for null, out-of-range, or unallocated primitives.
pub fn vertex_pull(ms: &MsInterface, ps_inputs: Option<&[SigElement]>) -> Result<String, &'static str> {
    let per_record = draw_vertices(ms, 1)?;
    let slots = ranked(&ms.outputs, true);
    let elems: Vec<(&SigElement, Option<usize>)> = match ps_inputs {
        None => slots.iter().enumerate().map(|(k, e)| (*e, Some(k))).collect(),
        Some(ins) => {
            let mut v: Vec<_> = ranked(ins, false).into_iter().map(|e| {
                let slot = slots.iter().position(|o| o.system_value == 0
                    && o.name.eq_ignore_ascii_case(&e.name) && o.index == e.index);
                (e, slot)
            }).collect();
            v.extend(slots.iter().enumerate().filter(|(_, o)| matches!(o.system_value, 2 | 3))
                .map(|(k, o)| (*o, Some(k))));
            v
        }
    };
    let mut h = String::from("RWByteAddressBuffer res : register(u1);\nstruct XclVert {\n    float4 pos : SV_Position;\n");
    let mut fill = String::new();
    for (k, (e, slot)) in elems.iter().enumerate() {
        let ty = match e.component_type { 1 => "uint", 2 => "int", _ => "float" };
        let n = if matches!(e.system_value, 2 | 3) {
            (8 - e.mask.leading_zeros()) as usize
        } else { 4 };
        if !(1..=4).contains(&n) { return Err("invalid vertex output mask"); }
        let decl = if n == 1 { ty.to_string() } else { format!("{ty}{n}") };
        let semantic = e.semantic();
        let _ = writeln!(h, "    {decl} p{k} : {semantic};");
        let sw = ["x", "y", "z", "w"][..n].concat();
        let raw = match slot {
            Some(s) => format!("res.Load4(b + {}u).{sw}", 16 * (s + 1)),
            None => format!("uint4(0u, 0u, 0u, 0u).{sw}"),
        };
        let val = match e.component_type {
            1 => raw, 2 => format!("asint({raw})"), _ => format!("asfloat({raw})"),
        };
        let _ = writeln!(fill, "    o.p{k} = {val};");
    }
    let rec = ms.record_dwords();
    let vd = ms.vert_dwords();
    let mv = ms.max_vertices;
    let mp = ms.max_primitives;
    let tri_base = MESH_HEADER + mv as usize * vd;
    h.push_str("};\nXclVert main(uint vertex : SV_VertexID) {\n    XclVert o = (XclVert)0;\n    o.pos = float4(2.0, 2.0, 2.0, 1.0);\n");
    let _ = writeln!(h, "    uint rec = (vertex / {per_record}u) * {rec}u;");
    let _ = writeln!(h, "    uint j = (vertex % {per_record}u) / 3u;");
    h.push_str("    uint nv = ");
    let _ = writeln!(h, "min(res.Load(rec * 4u), {mv}u);");
    let _ = writeln!(h, "    uint np = min(res.Load(rec * 4u + 4u), {mp}u);");
    h.push_str("    if (j >= np || nv == 0u) return o;\n");
    let _ = writeln!(h, "    uint p = res.Load((rec + {tri_base}u + j) * 4u);");
    h.push_str("    uint3 x = uint3(p & 0x3ffu, (p >> 10) & 0x3ffu, (p >> 20) & 0x3ffu);\n    if ((p >> 31) != 0u || any(x >= nv)) return o;\n");
    let _ = writeln!(h, "    uint b = (rec + {MESH_HEADER}u + x[vertex % 3u] * {vd}u) * 4u;");
    h.push_str("    o.pos = asfloat(res.Load4(b));\n");
    h.push_str(&fill);
    h.push_str("    return o;\n}\n");
    Ok(h)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sig(name: &str, system_value: u32, register: u32) -> SigElement {
        SigElement { name: name.into(), index: 0, system_value, component_type: 3,
            register, mask: 0xf, rw_mask: 0xf, stream: 0 }
    }

    fn mesh() -> MsInterface {
        MsInterface { outputs: vec![sig("SV_Position", 1, 0), sig("TEXCOORD", 0, 1)],
            threads: 128, max_vertices: 128, max_primitives: 128 }
    }

    #[test]
    fn fallback_draw_bounds() {
        let mut m = mesh();
        assert_eq!(draw_vertices(&m, 0), Ok(0));
        assert_eq!(draw_vertices(&m, 2), Ok(768));
        assert!(draw_vertices(&m, u32::MAX).is_err());
        m.max_primitives = 0;
        assert!(draw_vertices(&m, 1).is_err());
        assert!(vertex_pull(&m, None).is_err());
    }

    #[test]
    fn fallback_source_layout() {
        let m = mesh();
        let h = vertex_pull(&m, None).unwrap();
        assert!(h.contains("SV_VertexID") && h.contains("register(u1)"));
        assert!(h.contains("(vertex / 384u) * 1156u"));
        assert!(h.contains("rec + 1028u + j"));
        assert!(h.contains("any(x >= nv)") && h.contains("p >> 31"));
        assert!(h.contains("rec + 4u + x[vertex % 3u] * 8u"));
        assert!(h.contains("res.Load4(b + 16u)"));
    }

    #[test]
    fn fallback_vertex_shader_compiles() {
        // Deliberately require DXC: this test must not silently pass without it.
        let m = mesh();
        let inputs = [sig("TEXCOORD", 0, 1), sig("COLOR", 0, 2)];
        for h in [vertex_pull(&m, None).unwrap(), vertex_pull(&m, Some(&inputs)).unwrap()] {
            assert!(!super::super::dxc::compile(&h, "main", "vs_6_6", &[])
                .unwrap_or_else(|e| panic!("{e}\n{h}")).is_empty());
        }
    }

    #[test]
    fn fallback_gpu_rasterizes_records() {
        use super::super::{dxc, runner};
        let adapter = runner::adapter().expect("D3D12 adapter required");
        println!("mesh fallback adapter: {adapter}");
        assert!(!adapter.contains("WARP"), "hardware adapter required");
        let m = MsInterface {
            outputs: vec![sig("SV_Position", 1, 0)],
            threads: 1, max_vertices: 3, max_primitives: 3,
        };
        // Each record: 4 header dwords, 12 vertex dwords, 3 primitive dwords.
        // Group zero draws on the left; group one has only invalid primitives.
        let cs = r#"
RWByteAddressBuffer res : register(u1);
[numthreads(1, 1, 1)]
void main(uint3 gid : SV_GroupID) {
    uint b = gid.x * 76u;
    float x = gid.x == 0u ? -0.5 : 0.5;
    res.Store4(b, uint4(3u, 2u, 0u, 0u));
    res.Store4(b + 16u, asuint(float4(x - 0.4, -0.5, 0.5, 1.0)));
    res.Store4(b + 32u, asuint(float4(x + 0.4, -0.5, 0.5, 1.0)));
    res.Store4(b + 48u, asuint(float4(x, 0.5, 0.5, 1.0)));
    res.Store(b + 64u, gid.x == 0u ? 0x00200400u : 0x80200400u);
    res.Store(b + 68u, 0x00300400u);
    res.Store(b + 72u, 0x00200400u);
}
"#;
        let vs = vertex_pull(&m, None).unwrap();
        let ps = "float4 main() : SV_Target0 { return float4(1, 0, 0, 1); }";
        let cs = dxc::compile(cs, "main", "cs_6_6", &[]).unwrap();
        let vs = dxc::compile(&vs, "main", "vs_6_6", &[]).unwrap();
        let ps = dxc::compile(ps, "main", "ps_6_6", &[]).unwrap();
        let (pixels, records) = runner::mesh_vertex_draw(
            &cs, &vs, &ps, 2, &m, &[], &[], 64, [0.0, 0.0, 0.0, 1.0],
        ).unwrap();
        assert_eq!(records.len(), 38);
        assert_eq!(&records[0..2], &[3, 2]);
        assert_eq!(&records[19..21], &[3, 2]);
        assert_eq!(pixels.len(), 4096);
        assert_eq!(pixels[32 * 64 + 16], [1.0, 0.0, 0.0, 1.0]);
        assert_eq!(pixels[32 * 64 + 48], [0.0, 0.0, 0.0, 1.0]);
        let red = pixels.iter().filter(|p| p[0] > 0.5).count();
        assert!((250..400).contains(&red), "unexpected triangle coverage: {red}");
        for y in 0..64 {
            for x in 32..64 {
                assert_eq!(pixels[y * 64 + x], [0.0, 0.0, 0.0, 1.0],
                    "invalid triangle rendered at {x},{y}");
            }
        }
        println!("mesh-free GPU draw: {red} red pixels; invalid records clipped");
    }
}

use std::{
    cmp::Ordering,
    fs,
    path::Path,
};

use vc_core::{Result, VcError};

pub const IVF_MAGIC: &[u8; 8] = b"RVCIVF01";

#[derive(Debug, Clone)]
pub struct IvfIndex {
    nlist: usize,
    dims: usize,
    default_nprobe: usize,
    centroids: Vec<f32>,
    offsets: Vec<u64>,
    vectors: Vec<f32>,
    ntotal: usize,
}

impl IvfIndex {
    pub fn load(path: &Path) -> Result<Self> {
        let raw = fs::read(path).map_err(|e| {
            VcError::Config(format!(
                "failed to read IVF index file {}: {e}",
                path.display()
            ))
        })?;
        if raw.len() < 24 {
            return Err(VcError::Config(format!(
                "IVF index file is too small: {}",
                path.display()
            )));
        }
        if &raw[..8] != IVF_MAGIC {
            return Err(VcError::Config(format!(
                "IVF index magic mismatch: {}",
                path.display()
            )));
        }

        let mut cursor = 8usize;
        let nlist = read_u32(&raw, &mut cursor)? as usize;
        let rows = read_u32(&raw, &mut cursor)? as usize;
        let dims = read_u32(&raw, &mut cursor)? as usize;
        let default_nprobe = read_u32(&raw, &mut cursor)? as usize;

        if nlist == 0 || rows == 0 || dims == 0 {
            return Err(VcError::Config(format!(
                "IVF index header is invalid: nlist={} rows={} dims={} ({})",
                nlist,
                rows,
                dims,
                path.display()
            )));
        }

        let centroids_len = nlist
            .checked_mul(dims)
            .ok_or_else(|| VcError::Config("IVF centroid size overflow".to_string()))?;
        let centroids = read_f32_vec(&raw, &mut cursor, centroids_len)?;

        let offsets_len = nlist
            .checked_add(1)
            .ok_or_else(|| VcError::Config("IVF offsets size overflow".to_string()))?;
        let offsets = read_u64_vec(&raw, &mut cursor, offsets_len)?;

        let vectors_len = rows
            .checked_mul(dims)
            .ok_or_else(|| VcError::Config("IVF vectors size overflow".to_string()))?;
        let vectors = read_f32_vec(&raw, &mut cursor, vectors_len)?;
        if cursor != raw.len() {
            return Err(VcError::Config(format!(
                "IVF index has trailing bytes: {}",
                path.display()
            )));
        }

        if offsets.first().copied().unwrap_or(1) != 0 {
            return Err(VcError::Config(format!(
                "IVF offsets must start at 0: {}",
                path.display()
            )));
        }
        if offsets.last().copied().unwrap_or_default() != rows as u64 {
            return Err(VcError::Config(format!(
                "IVF offsets final value mismatch: expected={} actual={} ({})",
                rows,
                offsets.last().copied().unwrap_or_default(),
                path.display()
            )));
        }
        for pair in offsets.windows(2) {
            if pair[0] > pair[1] {
                return Err(VcError::Config(format!(
                    "IVF offsets must be monotonic: {}",
                    path.display()
                )));
            }
        }

        Ok(Self {
            nlist,
            dims,
            default_nprobe: default_nprobe.clamp(1, nlist),
            centroids,
            offsets,
            vectors,
            ntotal: rows,
        })
    }

    pub fn search(
        &self,
        query: &[f32],
        top_k: usize,
        nprobe: usize,
        max_rows: usize,
    ) -> Vec<(f32, usize)> {
        self.search_with_stats(query, top_k, nprobe, max_rows).0
    }

    pub fn search_with_stats(
        &self,
        query: &[f32],
        top_k: usize,
        nprobe: usize,
        _max_rows: usize,
    ) -> (Vec<(f32, usize)>, usize) {
        if self.dims == 0 || self.ntotal() == 0 || query.len() != self.dims || top_k == 0 {
            return (Vec::new(), 0);
        }

        let probe_count = nprobe.max(1).min(self.nlist);
        let mut centroid_best = Vec::<(f32, usize)>::with_capacity(probe_count);
        for cluster in 0..self.nlist {
            let centroid_start = cluster * self.dims;
            let centroid_end = centroid_start + self.dims;
            let dist = l2_distance(query, &self.centroids[centroid_start..centroid_end]);
            push_smallest(&mut centroid_best, (dist, cluster), probe_count);
        }
        centroid_best.sort_by(float_pair_asc);

        let mut best = Vec::<(f32, usize)>::with_capacity(top_k);
        let mut scanned_rows = 0usize;
        for &(_, cluster) in &centroid_best {
            let start = self.offsets[cluster] as usize;
            let end = self.offsets[cluster + 1] as usize;
            scanned_rows = scanned_rows.saturating_add(end.saturating_sub(start));
            for idx in start..end {
                let vec_start = idx * self.dims;
                let vec_end = vec_start + self.dims;
                let dist = l2_distance(query, &self.vectors[vec_start..vec_end]);
                push_smallest(&mut best, (dist, idx), top_k);
            }
        }

        best.sort_by(float_pair_asc);
        (best, scanned_rows)
    }

    pub fn ntotal(&self) -> usize {
        self.ntotal
    }

    pub fn nlist(&self) -> usize {
        self.nlist
    }

    pub fn default_nprobe(&self) -> usize {
        self.default_nprobe
    }

    pub fn dims(&self) -> usize {
        self.dims
    }

    pub fn vector(&self, idx: usize) -> Option<&[f32]> {
        let start = idx.checked_mul(self.dims)?;
        let end = start.checked_add(self.dims)?;
        self.vectors.get(start..end)
    }
}

fn read_u32(raw: &[u8], cursor: &mut usize) -> Result<u32> {
    let end = cursor
        .checked_add(4)
        .ok_or_else(|| VcError::Config("IVF u32 offset overflow".to_string()))?;
    let bytes = raw.get(*cursor..end).ok_or_else(|| {
        VcError::Config("IVF file truncated while reading u32".to_string())
    })?;
    *cursor = end;
    Ok(u32::from_le_bytes(bytes.try_into().unwrap()))
}

fn read_u64_vec(raw: &[u8], cursor: &mut usize, len: usize) -> Result<Vec<u64>> {
    let bytes_len = len
        .checked_mul(8)
        .ok_or_else(|| VcError::Config("IVF u64 vector size overflow".to_string()))?;
    let end = cursor
        .checked_add(bytes_len)
        .ok_or_else(|| VcError::Config("IVF u64 vector offset overflow".to_string()))?;
    let bytes = raw.get(*cursor..end).ok_or_else(|| {
        VcError::Config("IVF file truncated while reading offsets".to_string())
    })?;
    let mut out = Vec::with_capacity(len);
    for chunk in bytes.chunks_exact(8) {
        out.push(u64::from_le_bytes(chunk.try_into().unwrap()));
    }
    *cursor = end;
    Ok(out)
}

fn read_f32_vec(raw: &[u8], cursor: &mut usize, len: usize) -> Result<Vec<f32>> {
    let bytes_len = len
        .checked_mul(4)
        .ok_or_else(|| VcError::Config("IVF f32 vector size overflow".to_string()))?;
    let end = cursor
        .checked_add(bytes_len)
        .ok_or_else(|| VcError::Config("IVF f32 vector offset overflow".to_string()))?;
    let bytes = raw.get(*cursor..end).ok_or_else(|| {
        VcError::Config("IVF file truncated while reading float payload".to_string())
    })?;
    let mut out = Vec::with_capacity(len);
    for chunk in bytes.chunks_exact(4) {
        out.push(f32::from_le_bytes(chunk.try_into().unwrap()));
    }
    *cursor = end;
    Ok(out)
}

fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let len4 = a.len() / 4 * 4;
    let mut acc0 = 0.0_f32;
    let mut acc1 = 0.0_f32;
    let mut acc2 = 0.0_f32;
    let mut acc3 = 0.0_f32;

    let mut i = 0usize;
    while i < len4 {
        let d0 = a[i] - b[i];
        let d1 = a[i + 1] - b[i + 1];
        let d2 = a[i + 2] - b[i + 2];
        let d3 = a[i + 3] - b[i + 3];
        acc0 += d0 * d0;
        acc1 += d1 * d1;
        acc2 += d2 * d2;
        acc3 += d3 * d3;
        i += 4;
    }

    let mut tail = 0.0_f32;
    while i < a.len() {
        let d = a[i] - b[i];
        tail += d * d;
        i += 1;
    }

    acc0 + acc1 + acc2 + acc3 + tail
}

fn push_smallest(best: &mut Vec<(f32, usize)>, cand: (f32, usize), k: usize) {
    if k == 0 {
        return;
    }
    if best.len() < k {
        best.push(cand);
        return;
    }
    let mut worst_idx = 0usize;
    let mut worst_dist = best[0].0;
    for (i, &(dist, _)) in best.iter().enumerate().skip(1) {
        if dist > worst_dist {
            worst_dist = dist;
            worst_idx = i;
        }
    }
    if cand.0 < worst_dist {
        best[worst_idx] = cand;
    }
}

fn float_pair_asc(a: &(f32, usize), b: &(f32, usize)) -> Ordering {
    a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal)
}

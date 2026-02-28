use std::sync::Mutex;

use faiss::{
    index::ivf_flat::IVFFlatIndex,
    read_index,
    Index as FaissIndexTrait,
};
use vc_core::{Result, VcError};

#[derive(Debug)]
pub struct FaissIndex {
    index: Mutex<IVFFlatIndex>,
    dims: usize,
    ntotal: usize,
    nlist: usize,
}

impl FaissIndex {
    pub fn load(path: &str, nprobe: usize) -> Result<Self> {
        let index = read_index(path)
            .map_err(|e| VcError::Config(format!("faiss read_index failed ({path}): {e}")))?;
        let mut index = index
            .into_ivf_flat()
            .map_err(|e| VcError::Config(format!("faiss index is not IVF+Flat ({path}): {e}")))?;
        let dims = index.d() as usize;
        let ntotal = index.ntotal() as usize;
        let nlist = index.nlist() as usize;
        let clamped_nprobe = nprobe.max(1).min(nlist.max(1));
        index.set_nprobe(clamped_nprobe as u32);
        Ok(Self {
            index: Mutex::new(index),
            dims,
            ntotal,
            nlist,
        })
    }

    pub fn set_nprobe(&self, nprobe: usize) -> Result<u32> {
        let mut index = self
            .index
            .lock()
            .map_err(|_| VcError::Inference("faiss index mutex poisoned".to_string()))?;
        let clamped = nprobe.max(1).min(self.nlist.max(1));
        index.set_nprobe(clamped as u32);
        Ok(clamped as u32)
    }

    pub fn search(&self, query: &[f32], top_k: usize) -> Result<Vec<u32>> {
        if query.len() != self.dims {
            return Err(VcError::Inference(format!(
                "faiss query dim mismatch: got={} expected={}",
                query.len(),
                self.dims
            )));
        }
        if top_k == 0 {
            return Ok(Vec::new());
        }
        let mut index = self
            .index
            .lock()
            .map_err(|_| VcError::Inference("faiss index mutex poisoned".to_string()))?;
        let result = index
            .search(query, top_k)
            .map_err(|e| VcError::Inference(format!("faiss search failed: {e}")))?;
        let mut labels = Vec::with_capacity(result.labels.len());
        for label in result.labels {
            let Some(id) = label.get() else {
                continue;
            };
            let id = u32::try_from(id).map_err(|_| {
                VcError::Inference(format!("faiss label exceeds u32 range: {id}"))
            })?;
            labels.push(id);
        }
        Ok(labels)
    }

    pub fn ntotal(&self) -> usize {
        self.ntotal
    }

    pub fn dims(&self) -> usize {
        self.dims
    }

    pub fn nlist(&self) -> usize {
        self.nlist
    }
}

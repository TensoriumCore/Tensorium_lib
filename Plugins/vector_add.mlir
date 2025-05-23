module attributes {gpu.container_module} {
  gpu.module @kernels {
    gpu.func @vector_add(%a: memref<8xf32>, %b: memref<8xf32>, %out: memref<8xf32>) kernel {
      %id = gpu.thread_id x
      %va = memref.load %a[%id] : memref<8xf32>
      %vb = memref.load %b[%id] : memref<8xf32>
      %r = arith.addf %va, %vb : f32
      memref.store %r, %out[%id] : memref<8xf32>
      gpu.return
    }
  }
}

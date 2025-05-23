module attributes {gpu.container_module} {
  memref.global "private" constant @A : memref<8xf32> = dense<0.0>
  memref.global "private" constant @B : memref<8xf32> = dense<0.0>
  memref.global "private" constant @OUT : memref<8xf32> = dense<0.0>

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

  func.func @main() {
    %a = memref.get_global @A : memref<8xf32>
    %b = memref.get_global @B : memref<8xf32>
    %out = memref.get_global @OUT : memref<8xf32>

    %c1_i64 = arith.constant 1 : i64
    %c8_i64 = arith.constant 8 : i64
    %c1 = arith.index_cast %c1_i64 : i64 to index
    %c8 = arith.index_cast %c8_i64 : i64 to index

    gpu.launch_func @kernels::@vector_add
      blocks in (%c1, %c1, %c1)
      threads in (%c8, %c1, %c1)
      args(%a: memref<8xf32>, %b: memref<8xf32>, %out: memref<8xf32>)

    return
  }
}

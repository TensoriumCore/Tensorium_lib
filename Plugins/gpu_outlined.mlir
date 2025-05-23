#tbaa_root = #llvm.tbaa_root<id = "Simple C++ TBAA">
#tbaa_type_desc = #llvm.tbaa_type_desc<id = "vtable pointer", members = {<#tbaa_root, 0>}>
#tbaa_type_desc1 = #llvm.tbaa_type_desc<id = "omnipotent char", members = {<#tbaa_root, 0>}>
#tbaa_tag = #llvm.tbaa_tag<base_type = #tbaa_type_desc, access_type = #tbaa_type_desc, offset = 0>
#tbaa_tag1 = #llvm.tbaa_tag<base_type = #tbaa_type_desc1, access_type = #tbaa_type_desc1, offset = 0>
#tbaa_type_desc2 = #llvm.tbaa_type_desc<id = "long", members = {<#tbaa_type_desc1, 0>}>
#tbaa_type_desc3 = #llvm.tbaa_type_desc<id = "_ZTSSt13_Ios_Fmtflags", members = {<#tbaa_type_desc1, 0>}>
#tbaa_type_desc4 = #llvm.tbaa_type_desc<id = "_ZTSSt12_Ios_Iostate", members = {<#tbaa_type_desc1, 0>}>
#tbaa_type_desc5 = #llvm.tbaa_type_desc<id = "any pointer", members = {<#tbaa_type_desc1, 0>}>
#tbaa_type_desc6 = #llvm.tbaa_type_desc<id = "int", members = {<#tbaa_type_desc1, 0>}>
#tbaa_type_desc7 = #llvm.tbaa_type_desc<id = "_ZTSNSt8ios_base6_WordsE", members = {<#tbaa_type_desc5, 0>, <#tbaa_type_desc2, 8>}>
#tbaa_type_desc8 = #llvm.tbaa_type_desc<id = "_ZTSSt6locale", members = {<#tbaa_type_desc5, 0>}>
#tbaa_type_desc9 = #llvm.tbaa_type_desc<id = "_ZTSSt8ios_base", members = {<#tbaa_type_desc2, 8>, <#tbaa_type_desc2, 16>, <#tbaa_type_desc3, 24>, <#tbaa_type_desc4, 28>, <#tbaa_type_desc4, 32>, <#tbaa_type_desc5, 40>, <#tbaa_type_desc7, 48>, <#tbaa_type_desc1, 64>, <#tbaa_type_desc6, 192>, <#tbaa_type_desc5, 200>, <#tbaa_type_desc8, 208>}>
#tbaa_tag2 = #llvm.tbaa_tag<base_type = #tbaa_type_desc9, access_type = #tbaa_type_desc2, offset = 16>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = ""} {
  llvm.mlir.global external @_ZSt4cout() {addr_space = 0 : i32, alignment = 8 : i64} : !llvm.struct<"class.std::basic_ostream", (ptr, struct<"class.std::basic_ios", (struct<"class.std::ios_base", (ptr, i64, i64, i32, i32, i32, ptr, struct<"struct.std::ios_base::_Words", (ptr, i64)>, array<8 x struct<"struct.std::ios_base::_Words", (ptr, i64)>>, i32, ptr, struct<"class.std::locale", (ptr)>)>, ptr, i8, i8, ptr, ptr, ptr, ptr)>)>
  llvm.mlir.global private unnamed_addr constant @".str"("Result (mul): \00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @".str.1"("\0A\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @".str.2"("Result (div): \00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.func local_unnamed_addr @main() -> (i32 {llvm.noundef}) attributes {passthrough = ["mustprogress", "norecurse", "sspstrong", ["uwtable", "2"], ["min-legal-vector-width", "256"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "4"], ["target-cpu", "x86-64"], ["target-features", "+avx,+cmov,+crc32,+cx8,+fxsr,+mmx,+popcnt,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave"], ["tune-cpu", "generic"]]} {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.mlir.addressof @_ZSt4cout : !llvm.ptr
    %2 = llvm.mlir.constant("Result (mul): \00") : !llvm.array<15 x i8>
    %3 = llvm.mlir.addressof @".str" : !llvm.ptr
    %4 = llvm.mlir.constant(14 : i64) : i64
    %5 = llvm.mlir.constant(-24 : i64) : i64
    %6 = llvm.mlir.constant(0 : i32) : i32
    %7 = llvm.mlir.constant(0 : i64) : i64
    %8 = llvm.getelementptr inbounds %1[0, 1, 0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.std::basic_ostream", (ptr, struct<"class.std::basic_ios", (struct<"class.std::ios_base", (ptr, i64, i64, i32, i32, i32, ptr, struct<"struct.std::ios_base::_Words", (ptr, i64)>, array<8 x struct<"struct.std::ios_base::_Words", (ptr, i64)>>, i32, ptr, struct<"class.std::locale", (ptr)>)>, ptr, i8, i8, ptr, ptr, ptr, ptr)>)>
    %9 = llvm.mlir.constant(8 : i64) : i64
    %10 = llvm.mlir.constant(-6.300000e+01 : f64) : f64
    %11 = llvm.mlir.constant(1 : i64) : i64
    %12 = llvm.mlir.constant(32 : i8) : i8
    %13 = llvm.mlir.constant(2 : i32) : i32
    %14 = llvm.mlir.constant(-4.500000e+01 : f64) : f64
    %15 = llvm.mlir.constant(-2.700000e+01 : f64) : f64
    %16 = llvm.mlir.constant(-9.000000e+00 : f64) : f64
    %17 = llvm.mlir.constant(9.000000e+00 : f64) : f64
    %18 = llvm.mlir.constant(2.700000e+01 : f64) : f64
    %19 = llvm.mlir.constant(4.500000e+01 : f64) : f64
    %20 = llvm.mlir.constant(6.300000e+01 : f64) : f64
    %21 = llvm.mlir.constant("\0A\00") : !llvm.array<2 x i8>
    %22 = llvm.mlir.addressof @".str.1" : !llvm.ptr
    %23 = llvm.mlir.constant("Result (div): \00") : !llvm.array<15 x i8>
    %24 = llvm.mlir.addressof @".str.2" : !llvm.ptr
    %25 = llvm.mlir.constant(-7.875000e+00 : f64) : f64
    %26 = llvm.mlir.constant(-6.4285712242126465 : f64) : f64
    %27 = llvm.mlir.constant(-4.500000e+00 : f64) : f64
    %28 = llvm.mlir.constant(-1.7999999523162842 : f64) : f64
    %29 = llvm.mlir.constant(2.250000e+00 : f64) : f64
    %30 = llvm.mlir.constant(2.250000e+01 : f64) : f64
    %31 = llvm.alloca %0 x i8 {alignment = 1 : i64} : (i32) -> !llvm.ptr
    %32 = llvm.alloca %0 x i8 {alignment = 1 : i64} : (i32) -> !llvm.ptr
    %33 = llvm.call @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(%1, %3, %4) : (!llvm.ptr, !llvm.ptr, i64) -> !llvm.ptr
    %34 = llvm.load %1 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> !llvm.ptr
    %35 = llvm.getelementptr %34[-24] : (!llvm.ptr) -> !llvm.ptr, i8
    %36 = llvm.load %35 {alignment = 8 : i64} : !llvm.ptr -> i64
    %37 = llvm.getelementptr %8[%36] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %9, %37 {alignment = 8 : i64, tbaa = [#tbaa_tag2]} : i64, !llvm.ptr
    %38 = llvm.call @_ZNSo9_M_insertIdEERSoT_(%1, %10) : (!llvm.ptr, f64) -> !llvm.ptr
    llvm.intr.lifetime.start 1, %32 : !llvm.ptr
    llvm.store %12, %32 {alignment = 1 : i64, tbaa = [#tbaa_tag1]} : i8, !llvm.ptr
    %39 = llvm.load %38 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> !llvm.ptr
    %40 = llvm.getelementptr %39[-24] : (!llvm.ptr) -> !llvm.ptr, i8
    %41 = llvm.load %40 {alignment = 8 : i64} : !llvm.ptr -> i64
    %42 = llvm.getelementptr inbounds %38[%41] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %43 = llvm.getelementptr inbounds %42[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.std::ios_base", (ptr, i64, i64, i32, i32, i32, ptr, struct<"struct.std::ios_base::_Words", (ptr, i64)>, array<8 x struct<"struct.std::ios_base::_Words", (ptr, i64)>>, i32, ptr, struct<"class.std::locale", (ptr)>)>
    %44 = llvm.load %43 {alignment = 8 : i64, tbaa = [#tbaa_tag2]} : !llvm.ptr -> i64
    %45 = llvm.icmp "eq" %44, %7 : i64
    llvm.cond_br %45, ^bb2, ^bb1
  ^bb1:  // pred: ^bb0
    %46 = llvm.call @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(%38, %32, %11) : (!llvm.ptr, !llvm.ptr, i64) -> !llvm.ptr
    llvm.br ^bb3
  ^bb2:  // pred: ^bb0
    %47 = llvm.call @_ZNSo3putEc(%38, %12) : (!llvm.ptr, i8) -> !llvm.ptr
    llvm.br ^bb3
  ^bb3:  // 2 preds: ^bb1, ^bb2
    llvm.intr.lifetime.end 1, %32 : !llvm.ptr
    %48 = llvm.load %1 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> !llvm.ptr
    %49 = llvm.getelementptr %48[-24] : (!llvm.ptr) -> !llvm.ptr, i8
    %50 = llvm.load %49 {alignment = 8 : i64} : !llvm.ptr -> i64
    %51 = llvm.getelementptr %8[%50] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %9, %51 {alignment = 8 : i64, tbaa = [#tbaa_tag2]} : i64, !llvm.ptr
    %52 = llvm.call @_ZNSo9_M_insertIdEERSoT_(%1, %14) : (!llvm.ptr, f64) -> !llvm.ptr
    llvm.intr.lifetime.start 1, %32 : !llvm.ptr
    llvm.store %12, %32 {alignment = 1 : i64, tbaa = [#tbaa_tag1]} : i8, !llvm.ptr
    %53 = llvm.load %52 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> !llvm.ptr
    %54 = llvm.getelementptr %53[-24] : (!llvm.ptr) -> !llvm.ptr, i8
    %55 = llvm.load %54 {alignment = 8 : i64} : !llvm.ptr -> i64
    %56 = llvm.getelementptr inbounds %52[%55] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %57 = llvm.getelementptr inbounds %56[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.std::ios_base", (ptr, i64, i64, i32, i32, i32, ptr, struct<"struct.std::ios_base::_Words", (ptr, i64)>, array<8 x struct<"struct.std::ios_base::_Words", (ptr, i64)>>, i32, ptr, struct<"class.std::locale", (ptr)>)>
    %58 = llvm.load %57 {alignment = 8 : i64, tbaa = [#tbaa_tag2]} : !llvm.ptr -> i64
    %59 = llvm.icmp "eq" %58, %7 : i64
    llvm.cond_br %59, ^bb5, ^bb4
  ^bb4:  // pred: ^bb3
    %60 = llvm.call @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(%52, %32, %11) : (!llvm.ptr, !llvm.ptr, i64) -> !llvm.ptr
    llvm.br ^bb6
  ^bb5:  // pred: ^bb3
    %61 = llvm.call @_ZNSo3putEc(%52, %12) : (!llvm.ptr, i8) -> !llvm.ptr
    llvm.br ^bb6
  ^bb6:  // 2 preds: ^bb4, ^bb5
    llvm.intr.lifetime.end 1, %32 : !llvm.ptr
    %62 = llvm.load %1 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> !llvm.ptr
    %63 = llvm.getelementptr %62[-24] : (!llvm.ptr) -> !llvm.ptr, i8
    %64 = llvm.load %63 {alignment = 8 : i64} : !llvm.ptr -> i64
    %65 = llvm.getelementptr %8[%64] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %9, %65 {alignment = 8 : i64, tbaa = [#tbaa_tag2]} : i64, !llvm.ptr
    %66 = llvm.call @_ZNSo9_M_insertIdEERSoT_(%1, %15) : (!llvm.ptr, f64) -> !llvm.ptr
    llvm.intr.lifetime.start 1, %32 : !llvm.ptr
    llvm.store %12, %32 {alignment = 1 : i64, tbaa = [#tbaa_tag1]} : i8, !llvm.ptr
    %67 = llvm.load %66 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> !llvm.ptr
    %68 = llvm.getelementptr %67[-24] : (!llvm.ptr) -> !llvm.ptr, i8
    %69 = llvm.load %68 {alignment = 8 : i64} : !llvm.ptr -> i64
    %70 = llvm.getelementptr inbounds %66[%69] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %71 = llvm.getelementptr inbounds %70[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.std::ios_base", (ptr, i64, i64, i32, i32, i32, ptr, struct<"struct.std::ios_base::_Words", (ptr, i64)>, array<8 x struct<"struct.std::ios_base::_Words", (ptr, i64)>>, i32, ptr, struct<"class.std::locale", (ptr)>)>
    %72 = llvm.load %71 {alignment = 8 : i64, tbaa = [#tbaa_tag2]} : !llvm.ptr -> i64
    %73 = llvm.icmp "eq" %72, %7 : i64
    llvm.cond_br %73, ^bb8, ^bb7
  ^bb7:  // pred: ^bb6
    %74 = llvm.call @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(%66, %32, %11) : (!llvm.ptr, !llvm.ptr, i64) -> !llvm.ptr
    llvm.br ^bb9
  ^bb8:  // pred: ^bb6
    %75 = llvm.call @_ZNSo3putEc(%66, %12) : (!llvm.ptr, i8) -> !llvm.ptr
    llvm.br ^bb9
  ^bb9:  // 2 preds: ^bb7, ^bb8
    llvm.intr.lifetime.end 1, %32 : !llvm.ptr
    %76 = llvm.load %1 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> !llvm.ptr
    %77 = llvm.getelementptr %76[-24] : (!llvm.ptr) -> !llvm.ptr, i8
    %78 = llvm.load %77 {alignment = 8 : i64} : !llvm.ptr -> i64
    %79 = llvm.getelementptr %8[%78] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %9, %79 {alignment = 8 : i64, tbaa = [#tbaa_tag2]} : i64, !llvm.ptr
    %80 = llvm.call @_ZNSo9_M_insertIdEERSoT_(%1, %16) : (!llvm.ptr, f64) -> !llvm.ptr
    llvm.intr.lifetime.start 1, %32 : !llvm.ptr
    llvm.store %12, %32 {alignment = 1 : i64, tbaa = [#tbaa_tag1]} : i8, !llvm.ptr
    %81 = llvm.load %80 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> !llvm.ptr
    %82 = llvm.getelementptr %81[-24] : (!llvm.ptr) -> !llvm.ptr, i8
    %83 = llvm.load %82 {alignment = 8 : i64} : !llvm.ptr -> i64
    %84 = llvm.getelementptr inbounds %80[%83] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %85 = llvm.getelementptr inbounds %84[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.std::ios_base", (ptr, i64, i64, i32, i32, i32, ptr, struct<"struct.std::ios_base::_Words", (ptr, i64)>, array<8 x struct<"struct.std::ios_base::_Words", (ptr, i64)>>, i32, ptr, struct<"class.std::locale", (ptr)>)>
    %86 = llvm.load %85 {alignment = 8 : i64, tbaa = [#tbaa_tag2]} : !llvm.ptr -> i64
    %87 = llvm.icmp "eq" %86, %7 : i64
    llvm.cond_br %87, ^bb11, ^bb10
  ^bb10:  // pred: ^bb9
    %88 = llvm.call @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(%80, %32, %11) : (!llvm.ptr, !llvm.ptr, i64) -> !llvm.ptr
    llvm.br ^bb12
  ^bb11:  // pred: ^bb9
    %89 = llvm.call @_ZNSo3putEc(%80, %12) : (!llvm.ptr, i8) -> !llvm.ptr
    llvm.br ^bb12
  ^bb12:  // 2 preds: ^bb10, ^bb11
    llvm.intr.lifetime.end 1, %32 : !llvm.ptr
    %90 = llvm.load %1 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> !llvm.ptr
    %91 = llvm.getelementptr %90[-24] : (!llvm.ptr) -> !llvm.ptr, i8
    %92 = llvm.load %91 {alignment = 8 : i64} : !llvm.ptr -> i64
    %93 = llvm.getelementptr %8[%92] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %9, %93 {alignment = 8 : i64, tbaa = [#tbaa_tag2]} : i64, !llvm.ptr
    %94 = llvm.call @_ZNSo9_M_insertIdEERSoT_(%1, %17) : (!llvm.ptr, f64) -> !llvm.ptr
    llvm.intr.lifetime.start 1, %32 : !llvm.ptr
    llvm.store %12, %32 {alignment = 1 : i64, tbaa = [#tbaa_tag1]} : i8, !llvm.ptr
    %95 = llvm.load %94 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> !llvm.ptr
    %96 = llvm.getelementptr %95[-24] : (!llvm.ptr) -> !llvm.ptr, i8
    %97 = llvm.load %96 {alignment = 8 : i64} : !llvm.ptr -> i64
    %98 = llvm.getelementptr inbounds %94[%97] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %99 = llvm.getelementptr inbounds %98[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.std::ios_base", (ptr, i64, i64, i32, i32, i32, ptr, struct<"struct.std::ios_base::_Words", (ptr, i64)>, array<8 x struct<"struct.std::ios_base::_Words", (ptr, i64)>>, i32, ptr, struct<"class.std::locale", (ptr)>)>
    %100 = llvm.load %99 {alignment = 8 : i64, tbaa = [#tbaa_tag2]} : !llvm.ptr -> i64
    %101 = llvm.icmp "eq" %100, %7 : i64
    llvm.cond_br %101, ^bb14, ^bb13
  ^bb13:  // pred: ^bb12
    %102 = llvm.call @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(%94, %32, %11) : (!llvm.ptr, !llvm.ptr, i64) -> !llvm.ptr
    llvm.br ^bb15
  ^bb14:  // pred: ^bb12
    %103 = llvm.call @_ZNSo3putEc(%94, %12) : (!llvm.ptr, i8) -> !llvm.ptr
    llvm.br ^bb15
  ^bb15:  // 2 preds: ^bb13, ^bb14
    llvm.intr.lifetime.end 1, %32 : !llvm.ptr
    %104 = llvm.load %1 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> !llvm.ptr
    %105 = llvm.getelementptr %104[-24] : (!llvm.ptr) -> !llvm.ptr, i8
    %106 = llvm.load %105 {alignment = 8 : i64} : !llvm.ptr -> i64
    %107 = llvm.getelementptr %8[%106] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %9, %107 {alignment = 8 : i64, tbaa = [#tbaa_tag2]} : i64, !llvm.ptr
    %108 = llvm.call @_ZNSo9_M_insertIdEERSoT_(%1, %18) : (!llvm.ptr, f64) -> !llvm.ptr
    llvm.intr.lifetime.start 1, %32 : !llvm.ptr
    llvm.store %12, %32 {alignment = 1 : i64, tbaa = [#tbaa_tag1]} : i8, !llvm.ptr
    %109 = llvm.load %108 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> !llvm.ptr
    %110 = llvm.getelementptr %109[-24] : (!llvm.ptr) -> !llvm.ptr, i8
    %111 = llvm.load %110 {alignment = 8 : i64} : !llvm.ptr -> i64
    %112 = llvm.getelementptr inbounds %108[%111] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %113 = llvm.getelementptr inbounds %112[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.std::ios_base", (ptr, i64, i64, i32, i32, i32, ptr, struct<"struct.std::ios_base::_Words", (ptr, i64)>, array<8 x struct<"struct.std::ios_base::_Words", (ptr, i64)>>, i32, ptr, struct<"class.std::locale", (ptr)>)>
    %114 = llvm.load %113 {alignment = 8 : i64, tbaa = [#tbaa_tag2]} : !llvm.ptr -> i64
    %115 = llvm.icmp "eq" %114, %7 : i64
    llvm.cond_br %115, ^bb17, ^bb16
  ^bb16:  // pred: ^bb15
    %116 = llvm.call @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(%108, %32, %11) : (!llvm.ptr, !llvm.ptr, i64) -> !llvm.ptr
    llvm.br ^bb18
  ^bb17:  // pred: ^bb15
    %117 = llvm.call @_ZNSo3putEc(%108, %12) : (!llvm.ptr, i8) -> !llvm.ptr
    llvm.br ^bb18
  ^bb18:  // 2 preds: ^bb16, ^bb17
    llvm.intr.lifetime.end 1, %32 : !llvm.ptr
    %118 = llvm.load %1 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> !llvm.ptr
    %119 = llvm.getelementptr %118[-24] : (!llvm.ptr) -> !llvm.ptr, i8
    %120 = llvm.load %119 {alignment = 8 : i64} : !llvm.ptr -> i64
    %121 = llvm.getelementptr %8[%120] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %9, %121 {alignment = 8 : i64, tbaa = [#tbaa_tag2]} : i64, !llvm.ptr
    %122 = llvm.call @_ZNSo9_M_insertIdEERSoT_(%1, %19) : (!llvm.ptr, f64) -> !llvm.ptr
    llvm.intr.lifetime.start 1, %32 : !llvm.ptr
    llvm.store %12, %32 {alignment = 1 : i64, tbaa = [#tbaa_tag1]} : i8, !llvm.ptr
    %123 = llvm.load %122 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> !llvm.ptr
    %124 = llvm.getelementptr %123[-24] : (!llvm.ptr) -> !llvm.ptr, i8
    %125 = llvm.load %124 {alignment = 8 : i64} : !llvm.ptr -> i64
    %126 = llvm.getelementptr inbounds %122[%125] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %127 = llvm.getelementptr inbounds %126[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.std::ios_base", (ptr, i64, i64, i32, i32, i32, ptr, struct<"struct.std::ios_base::_Words", (ptr, i64)>, array<8 x struct<"struct.std::ios_base::_Words", (ptr, i64)>>, i32, ptr, struct<"class.std::locale", (ptr)>)>
    %128 = llvm.load %127 {alignment = 8 : i64, tbaa = [#tbaa_tag2]} : !llvm.ptr -> i64
    %129 = llvm.icmp "eq" %128, %7 : i64
    llvm.cond_br %129, ^bb20, ^bb19
  ^bb19:  // pred: ^bb18
    %130 = llvm.call @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(%122, %32, %11) : (!llvm.ptr, !llvm.ptr, i64) -> !llvm.ptr
    llvm.br ^bb21
  ^bb20:  // pred: ^bb18
    %131 = llvm.call @_ZNSo3putEc(%122, %12) : (!llvm.ptr, i8) -> !llvm.ptr
    llvm.br ^bb21
  ^bb21:  // 2 preds: ^bb19, ^bb20
    llvm.intr.lifetime.end 1, %32 : !llvm.ptr
    %132 = llvm.load %1 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> !llvm.ptr
    %133 = llvm.getelementptr %132[-24] : (!llvm.ptr) -> !llvm.ptr, i8
    %134 = llvm.load %133 {alignment = 8 : i64} : !llvm.ptr -> i64
    %135 = llvm.getelementptr %8[%134] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %9, %135 {alignment = 8 : i64, tbaa = [#tbaa_tag2]} : i64, !llvm.ptr
    %136 = llvm.call @_ZNSo9_M_insertIdEERSoT_(%1, %20) : (!llvm.ptr, f64) -> !llvm.ptr
    llvm.intr.lifetime.start 1, %32 : !llvm.ptr
    llvm.store %12, %32 {alignment = 1 : i64, tbaa = [#tbaa_tag1]} : i8, !llvm.ptr
    %137 = llvm.load %136 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> !llvm.ptr
    %138 = llvm.getelementptr %137[-24] : (!llvm.ptr) -> !llvm.ptr, i8
    %139 = llvm.load %138 {alignment = 8 : i64} : !llvm.ptr -> i64
    %140 = llvm.getelementptr inbounds %136[%139] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %141 = llvm.getelementptr inbounds %140[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.std::ios_base", (ptr, i64, i64, i32, i32, i32, ptr, struct<"struct.std::ios_base::_Words", (ptr, i64)>, array<8 x struct<"struct.std::ios_base::_Words", (ptr, i64)>>, i32, ptr, struct<"class.std::locale", (ptr)>)>
    %142 = llvm.load %141 {alignment = 8 : i64, tbaa = [#tbaa_tag2]} : !llvm.ptr -> i64
    %143 = llvm.icmp "eq" %142, %7 : i64
    llvm.cond_br %143, ^bb23, ^bb22
  ^bb22:  // pred: ^bb21
    %144 = llvm.call @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(%136, %32, %11) : (!llvm.ptr, !llvm.ptr, i64) -> !llvm.ptr
    llvm.br ^bb24
  ^bb23:  // pred: ^bb21
    %145 = llvm.call @_ZNSo3putEc(%136, %12) : (!llvm.ptr, i8) -> !llvm.ptr
    llvm.br ^bb24
  ^bb24:  // 2 preds: ^bb22, ^bb23
    llvm.intr.lifetime.end 1, %32 : !llvm.ptr
    %146 = llvm.call @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(%1, %22, %11) : (!llvm.ptr, !llvm.ptr, i64) -> !llvm.ptr
    %147 = llvm.call @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(%1, %24, %4) : (!llvm.ptr, !llvm.ptr, i64) -> !llvm.ptr
    %148 = llvm.load %1 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> !llvm.ptr
    %149 = llvm.getelementptr %148[-24] : (!llvm.ptr) -> !llvm.ptr, i8
    %150 = llvm.load %149 {alignment = 8 : i64} : !llvm.ptr -> i64
    %151 = llvm.getelementptr %8[%150] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %9, %151 {alignment = 8 : i64, tbaa = [#tbaa_tag2]} : i64, !llvm.ptr
    %152 = llvm.call @_ZNSo9_M_insertIdEERSoT_(%1, %25) : (!llvm.ptr, f64) -> !llvm.ptr
    llvm.intr.lifetime.start 1, %31 : !llvm.ptr
    llvm.store %12, %31 {alignment = 1 : i64, tbaa = [#tbaa_tag1]} : i8, !llvm.ptr
    %153 = llvm.load %152 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> !llvm.ptr
    %154 = llvm.getelementptr %153[-24] : (!llvm.ptr) -> !llvm.ptr, i8
    %155 = llvm.load %154 {alignment = 8 : i64} : !llvm.ptr -> i64
    %156 = llvm.getelementptr inbounds %152[%155] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %157 = llvm.getelementptr inbounds %156[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.std::ios_base", (ptr, i64, i64, i32, i32, i32, ptr, struct<"struct.std::ios_base::_Words", (ptr, i64)>, array<8 x struct<"struct.std::ios_base::_Words", (ptr, i64)>>, i32, ptr, struct<"class.std::locale", (ptr)>)>
    %158 = llvm.load %157 {alignment = 8 : i64, tbaa = [#tbaa_tag2]} : !llvm.ptr -> i64
    %159 = llvm.icmp "eq" %158, %7 : i64
    llvm.cond_br %159, ^bb26, ^bb25
  ^bb25:  // pred: ^bb24
    %160 = llvm.call @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(%152, %31, %11) : (!llvm.ptr, !llvm.ptr, i64) -> !llvm.ptr
    llvm.br ^bb27
  ^bb26:  // pred: ^bb24
    %161 = llvm.call @_ZNSo3putEc(%152, %12) : (!llvm.ptr, i8) -> !llvm.ptr
    llvm.br ^bb27
  ^bb27:  // 2 preds: ^bb25, ^bb26
    llvm.intr.lifetime.end 1, %31 : !llvm.ptr
    %162 = llvm.load %1 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> !llvm.ptr
    %163 = llvm.getelementptr %162[-24] : (!llvm.ptr) -> !llvm.ptr, i8
    %164 = llvm.load %163 {alignment = 8 : i64} : !llvm.ptr -> i64
    %165 = llvm.getelementptr %8[%164] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %9, %165 {alignment = 8 : i64, tbaa = [#tbaa_tag2]} : i64, !llvm.ptr
    %166 = llvm.call @_ZNSo9_M_insertIdEERSoT_(%1, %26) : (!llvm.ptr, f64) -> !llvm.ptr
    llvm.intr.lifetime.start 1, %31 : !llvm.ptr
    llvm.store %12, %31 {alignment = 1 : i64, tbaa = [#tbaa_tag1]} : i8, !llvm.ptr
    %167 = llvm.load %166 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> !llvm.ptr
    %168 = llvm.getelementptr %167[-24] : (!llvm.ptr) -> !llvm.ptr, i8
    %169 = llvm.load %168 {alignment = 8 : i64} : !llvm.ptr -> i64
    %170 = llvm.getelementptr inbounds %166[%169] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %171 = llvm.getelementptr inbounds %170[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.std::ios_base", (ptr, i64, i64, i32, i32, i32, ptr, struct<"struct.std::ios_base::_Words", (ptr, i64)>, array<8 x struct<"struct.std::ios_base::_Words", (ptr, i64)>>, i32, ptr, struct<"class.std::locale", (ptr)>)>
    %172 = llvm.load %171 {alignment = 8 : i64, tbaa = [#tbaa_tag2]} : !llvm.ptr -> i64
    %173 = llvm.icmp "eq" %172, %7 : i64
    llvm.cond_br %173, ^bb29, ^bb28
  ^bb28:  // pred: ^bb27
    %174 = llvm.call @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(%166, %31, %11) : (!llvm.ptr, !llvm.ptr, i64) -> !llvm.ptr
    llvm.br ^bb30
  ^bb29:  // pred: ^bb27
    %175 = llvm.call @_ZNSo3putEc(%166, %12) : (!llvm.ptr, i8) -> !llvm.ptr
    llvm.br ^bb30
  ^bb30:  // 2 preds: ^bb28, ^bb29
    llvm.intr.lifetime.end 1, %31 : !llvm.ptr
    %176 = llvm.load %1 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> !llvm.ptr
    %177 = llvm.getelementptr %176[-24] : (!llvm.ptr) -> !llvm.ptr, i8
    %178 = llvm.load %177 {alignment = 8 : i64} : !llvm.ptr -> i64
    %179 = llvm.getelementptr %8[%178] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %9, %179 {alignment = 8 : i64, tbaa = [#tbaa_tag2]} : i64, !llvm.ptr
    %180 = llvm.call @_ZNSo9_M_insertIdEERSoT_(%1, %27) : (!llvm.ptr, f64) -> !llvm.ptr
    llvm.intr.lifetime.start 1, %31 : !llvm.ptr
    llvm.store %12, %31 {alignment = 1 : i64, tbaa = [#tbaa_tag1]} : i8, !llvm.ptr
    %181 = llvm.load %180 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> !llvm.ptr
    %182 = llvm.getelementptr %181[-24] : (!llvm.ptr) -> !llvm.ptr, i8
    %183 = llvm.load %182 {alignment = 8 : i64} : !llvm.ptr -> i64
    %184 = llvm.getelementptr inbounds %180[%183] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %185 = llvm.getelementptr inbounds %184[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.std::ios_base", (ptr, i64, i64, i32, i32, i32, ptr, struct<"struct.std::ios_base::_Words", (ptr, i64)>, array<8 x struct<"struct.std::ios_base::_Words", (ptr, i64)>>, i32, ptr, struct<"class.std::locale", (ptr)>)>
    %186 = llvm.load %185 {alignment = 8 : i64, tbaa = [#tbaa_tag2]} : !llvm.ptr -> i64
    %187 = llvm.icmp "eq" %186, %7 : i64
    llvm.cond_br %187, ^bb32, ^bb31
  ^bb31:  // pred: ^bb30
    %188 = llvm.call @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(%180, %31, %11) : (!llvm.ptr, !llvm.ptr, i64) -> !llvm.ptr
    llvm.br ^bb33
  ^bb32:  // pred: ^bb30
    %189 = llvm.call @_ZNSo3putEc(%180, %12) : (!llvm.ptr, i8) -> !llvm.ptr
    llvm.br ^bb33
  ^bb33:  // 2 preds: ^bb31, ^bb32
    llvm.intr.lifetime.end 1, %31 : !llvm.ptr
    %190 = llvm.load %1 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> !llvm.ptr
    %191 = llvm.getelementptr %190[-24] : (!llvm.ptr) -> !llvm.ptr, i8
    %192 = llvm.load %191 {alignment = 8 : i64} : !llvm.ptr -> i64
    %193 = llvm.getelementptr %8[%192] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %9, %193 {alignment = 8 : i64, tbaa = [#tbaa_tag2]} : i64, !llvm.ptr
    %194 = llvm.call @_ZNSo9_M_insertIdEERSoT_(%1, %28) : (!llvm.ptr, f64) -> !llvm.ptr
    llvm.intr.lifetime.start 1, %31 : !llvm.ptr
    llvm.store %12, %31 {alignment = 1 : i64, tbaa = [#tbaa_tag1]} : i8, !llvm.ptr
    %195 = llvm.load %194 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> !llvm.ptr
    %196 = llvm.getelementptr %195[-24] : (!llvm.ptr) -> !llvm.ptr, i8
    %197 = llvm.load %196 {alignment = 8 : i64} : !llvm.ptr -> i64
    %198 = llvm.getelementptr inbounds %194[%197] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %199 = llvm.getelementptr inbounds %198[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.std::ios_base", (ptr, i64, i64, i32, i32, i32, ptr, struct<"struct.std::ios_base::_Words", (ptr, i64)>, array<8 x struct<"struct.std::ios_base::_Words", (ptr, i64)>>, i32, ptr, struct<"class.std::locale", (ptr)>)>
    %200 = llvm.load %199 {alignment = 8 : i64, tbaa = [#tbaa_tag2]} : !llvm.ptr -> i64
    %201 = llvm.icmp "eq" %200, %7 : i64
    llvm.cond_br %201, ^bb35, ^bb34
  ^bb34:  // pred: ^bb33
    %202 = llvm.call @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(%194, %31, %11) : (!llvm.ptr, !llvm.ptr, i64) -> !llvm.ptr
    llvm.br ^bb36
  ^bb35:  // pred: ^bb33
    %203 = llvm.call @_ZNSo3putEc(%194, %12) : (!llvm.ptr, i8) -> !llvm.ptr
    llvm.br ^bb36
  ^bb36:  // 2 preds: ^bb34, ^bb35
    llvm.intr.lifetime.end 1, %31 : !llvm.ptr
    %204 = llvm.load %1 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> !llvm.ptr
    %205 = llvm.getelementptr %204[-24] : (!llvm.ptr) -> !llvm.ptr, i8
    %206 = llvm.load %205 {alignment = 8 : i64} : !llvm.ptr -> i64
    %207 = llvm.getelementptr %8[%206] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %9, %207 {alignment = 8 : i64, tbaa = [#tbaa_tag2]} : i64, !llvm.ptr
    %208 = llvm.call @_ZNSo9_M_insertIdEERSoT_(%1, %29) : (!llvm.ptr, f64) -> !llvm.ptr
    llvm.intr.lifetime.start 1, %31 : !llvm.ptr
    llvm.store %12, %31 {alignment = 1 : i64, tbaa = [#tbaa_tag1]} : i8, !llvm.ptr
    %209 = llvm.load %208 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> !llvm.ptr
    %210 = llvm.getelementptr %209[-24] : (!llvm.ptr) -> !llvm.ptr, i8
    %211 = llvm.load %210 {alignment = 8 : i64} : !llvm.ptr -> i64
    %212 = llvm.getelementptr inbounds %208[%211] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %213 = llvm.getelementptr inbounds %212[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.std::ios_base", (ptr, i64, i64, i32, i32, i32, ptr, struct<"struct.std::ios_base::_Words", (ptr, i64)>, array<8 x struct<"struct.std::ios_base::_Words", (ptr, i64)>>, i32, ptr, struct<"class.std::locale", (ptr)>)>
    %214 = llvm.load %213 {alignment = 8 : i64, tbaa = [#tbaa_tag2]} : !llvm.ptr -> i64
    %215 = llvm.icmp "eq" %214, %7 : i64
    llvm.cond_br %215, ^bb38, ^bb37
  ^bb37:  // pred: ^bb36
    %216 = llvm.call @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(%208, %31, %11) : (!llvm.ptr, !llvm.ptr, i64) -> !llvm.ptr
    llvm.br ^bb39
  ^bb38:  // pred: ^bb36
    %217 = llvm.call @_ZNSo3putEc(%208, %12) : (!llvm.ptr, i8) -> !llvm.ptr
    llvm.br ^bb39
  ^bb39:  // 2 preds: ^bb37, ^bb38
    llvm.intr.lifetime.end 1, %31 : !llvm.ptr
    %218 = llvm.load %1 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> !llvm.ptr
    %219 = llvm.getelementptr %218[-24] : (!llvm.ptr) -> !llvm.ptr, i8
    %220 = llvm.load %219 {alignment = 8 : i64} : !llvm.ptr -> i64
    %221 = llvm.getelementptr %8[%220] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %9, %221 {alignment = 8 : i64, tbaa = [#tbaa_tag2]} : i64, !llvm.ptr
    %222 = llvm.call @_ZNSo9_M_insertIdEERSoT_(%1, %17) : (!llvm.ptr, f64) -> !llvm.ptr
    llvm.intr.lifetime.start 1, %31 : !llvm.ptr
    llvm.store %12, %31 {alignment = 1 : i64, tbaa = [#tbaa_tag1]} : i8, !llvm.ptr
    %223 = llvm.load %222 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> !llvm.ptr
    %224 = llvm.getelementptr %223[-24] : (!llvm.ptr) -> !llvm.ptr, i8
    %225 = llvm.load %224 {alignment = 8 : i64} : !llvm.ptr -> i64
    %226 = llvm.getelementptr inbounds %222[%225] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %227 = llvm.getelementptr inbounds %226[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.std::ios_base", (ptr, i64, i64, i32, i32, i32, ptr, struct<"struct.std::ios_base::_Words", (ptr, i64)>, array<8 x struct<"struct.std::ios_base::_Words", (ptr, i64)>>, i32, ptr, struct<"class.std::locale", (ptr)>)>
    %228 = llvm.load %227 {alignment = 8 : i64, tbaa = [#tbaa_tag2]} : !llvm.ptr -> i64
    %229 = llvm.icmp "eq" %228, %7 : i64
    llvm.cond_br %229, ^bb41, ^bb40
  ^bb40:  // pred: ^bb39
    %230 = llvm.call @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(%222, %31, %11) : (!llvm.ptr, !llvm.ptr, i64) -> !llvm.ptr
    llvm.br ^bb42
  ^bb41:  // pred: ^bb39
    %231 = llvm.call @_ZNSo3putEc(%222, %12) : (!llvm.ptr, i8) -> !llvm.ptr
    llvm.br ^bb42
  ^bb42:  // 2 preds: ^bb40, ^bb41
    llvm.intr.lifetime.end 1, %31 : !llvm.ptr
    %232 = llvm.load %1 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> !llvm.ptr
    %233 = llvm.getelementptr %232[-24] : (!llvm.ptr) -> !llvm.ptr, i8
    %234 = llvm.load %233 {alignment = 8 : i64} : !llvm.ptr -> i64
    %235 = llvm.getelementptr %8[%234] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %9, %235 {alignment = 8 : i64, tbaa = [#tbaa_tag2]} : i64, !llvm.ptr
    %236 = llvm.call @_ZNSo9_M_insertIdEERSoT_(%1, %30) : (!llvm.ptr, f64) -> !llvm.ptr
    llvm.intr.lifetime.start 1, %31 : !llvm.ptr
    llvm.store %12, %31 {alignment = 1 : i64, tbaa = [#tbaa_tag1]} : i8, !llvm.ptr
    %237 = llvm.load %236 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> !llvm.ptr
    %238 = llvm.getelementptr %237[-24] : (!llvm.ptr) -> !llvm.ptr, i8
    %239 = llvm.load %238 {alignment = 8 : i64} : !llvm.ptr -> i64
    %240 = llvm.getelementptr inbounds %236[%239] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %241 = llvm.getelementptr inbounds %240[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.std::ios_base", (ptr, i64, i64, i32, i32, i32, ptr, struct<"struct.std::ios_base::_Words", (ptr, i64)>, array<8 x struct<"struct.std::ios_base::_Words", (ptr, i64)>>, i32, ptr, struct<"class.std::locale", (ptr)>)>
    %242 = llvm.load %241 {alignment = 8 : i64, tbaa = [#tbaa_tag2]} : !llvm.ptr -> i64
    %243 = llvm.icmp "eq" %242, %7 : i64
    llvm.cond_br %243, ^bb44, ^bb43
  ^bb43:  // pred: ^bb42
    %244 = llvm.call @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(%236, %31, %11) : (!llvm.ptr, !llvm.ptr, i64) -> !llvm.ptr
    llvm.br ^bb45
  ^bb44:  // pred: ^bb42
    %245 = llvm.call @_ZNSo3putEc(%236, %12) : (!llvm.ptr, i8) -> !llvm.ptr
    llvm.br ^bb45
  ^bb45:  // 2 preds: ^bb43, ^bb44
    llvm.intr.lifetime.end 1, %31 : !llvm.ptr
    %246 = llvm.load %1 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> !llvm.ptr
    %247 = llvm.getelementptr %246[-24] : (!llvm.ptr) -> !llvm.ptr, i8
    %248 = llvm.load %247 {alignment = 8 : i64} : !llvm.ptr -> i64
    %249 = llvm.getelementptr %8[%248] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %9, %249 {alignment = 8 : i64, tbaa = [#tbaa_tag2]} : i64, !llvm.ptr
    %250 = llvm.call @_ZNSo9_M_insertIdEERSoT_(%1, %20) : (!llvm.ptr, f64) -> !llvm.ptr
    llvm.intr.lifetime.start 1, %31 : !llvm.ptr
    llvm.store %12, %31 {alignment = 1 : i64, tbaa = [#tbaa_tag1]} : i8, !llvm.ptr
    %251 = llvm.load %250 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> !llvm.ptr
    %252 = llvm.getelementptr %251[-24] : (!llvm.ptr) -> !llvm.ptr, i8
    %253 = llvm.load %252 {alignment = 8 : i64} : !llvm.ptr -> i64
    %254 = llvm.getelementptr inbounds %250[%253] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %255 = llvm.getelementptr inbounds %254[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.std::ios_base", (ptr, i64, i64, i32, i32, i32, ptr, struct<"struct.std::ios_base::_Words", (ptr, i64)>, array<8 x struct<"struct.std::ios_base::_Words", (ptr, i64)>>, i32, ptr, struct<"class.std::locale", (ptr)>)>
    %256 = llvm.load %255 {alignment = 8 : i64, tbaa = [#tbaa_tag2]} : !llvm.ptr -> i64
    %257 = llvm.icmp "eq" %256, %7 : i64
    llvm.cond_br %257, ^bb47, ^bb46
  ^bb46:  // pred: ^bb45
    %258 = llvm.call @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(%250, %31, %11) : (!llvm.ptr, !llvm.ptr, i64) -> !llvm.ptr
    llvm.br ^bb48
  ^bb47:  // pred: ^bb45
    %259 = llvm.call @_ZNSo3putEc(%250, %12) : (!llvm.ptr, i8) -> !llvm.ptr
    llvm.br ^bb48
  ^bb48:  // 2 preds: ^bb46, ^bb47
    llvm.intr.lifetime.end 1, %31 : !llvm.ptr
    %260 = llvm.call @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(%1, %22, %11) : (!llvm.ptr, !llvm.ptr, i64) -> !llvm.ptr
    llvm.return %6 : i32
  }
  llvm.func local_unnamed_addr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 8 : i64, llvm.nonnull, llvm.noundef}, !llvm.ptr {llvm.noundef}, i64 {llvm.noundef}) -> (!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 8 : i64, llvm.nonnull, llvm.noundef}) attributes {passthrough = [["no-trapping-math", "true"], ["stack-protector-buffer-size", "4"], ["target-cpu", "x86-64"], ["target-features", "+avx,+cmov,+crc32,+cx8,+fxsr,+mmx,+popcnt,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave"], ["tune-cpu", "generic"]]}
  llvm.func local_unnamed_addr @_ZNSo9_M_insertIdEERSoT_(!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 8 : i64, llvm.nonnull, llvm.noundef}, f64 {llvm.noundef}) -> (!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 8 : i64, llvm.nonnull, llvm.noundef}) attributes {passthrough = [["no-trapping-math", "true"], ["stack-protector-buffer-size", "4"], ["target-cpu", "x86-64"], ["target-features", "+avx,+cmov,+crc32,+cx8,+fxsr,+mmx,+popcnt,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave"], ["tune-cpu", "generic"]]}
  llvm.func local_unnamed_addr @_ZNSo3putEc(!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 8 : i64, llvm.nonnull, llvm.noundef}, i8 {llvm.noundef, llvm.signext}) -> (!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 8 : i64, llvm.nonnull, llvm.noundef}) attributes {passthrough = [["no-trapping-math", "true"], ["stack-protector-buffer-size", "4"], ["target-cpu", "x86-64"], ["target-features", "+avx,+cmov,+crc32,+cx8,+fxsr,+mmx,+popcnt,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave"], ["tune-cpu", "generic"]]}
}


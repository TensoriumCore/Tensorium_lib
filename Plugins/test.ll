; ModuleID = 'Plugins/test.cpp'
source_filename = "Plugins/test.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: mustprogress noinline optnone sspstrong uwtable
define noundef ptr @_Z1fv() #0 {
  %1 = alloca ptr, align 8
  %2 = call noalias noundef nonnull ptr @_Znam(i64 noundef 32) #3
  store ptr %2, ptr %1, align 8
  %3 = load ptr, ptr %1, align 8
  %4 = getelementptr float, ptr %3, i64 0
  store float 1.000000e+00, ptr %4, align 4
  %5 = load ptr, ptr %1, align 8
  ret ptr %5
}

; Function Attrs: nobuiltin allocsize(0)
declare noundef nonnull ptr @_Znam(i64 noundef) #1

; Function Attrs: mustprogress noinline norecurse optnone sspstrong uwtable
define noundef i32 @main() #2 {
  %1 = alloca i32, align 4
  %2 = alloca i32, align 4
  %3 = alloca [10 x float], align 16
  %4 = alloca [10 x float], align 16
  %5 = alloca i32, align 4
  %6 = alloca float, align 4
  store i32 0, ptr %1, align 4
  store i32 10, ptr %2, align 4
  store i32 0, ptr %5, align 4
  br label %7

7:                                                ; preds = %23, %0
  %8 = load i32, ptr %5, align 4
  %9 = icmp slt i32 %8, 10
  br i1 %9, label %10, label %26

10:                                               ; preds = %7
  %11 = load i32, ptr %5, align 4
  %12 = sitofp i32 %11 to float
  %13 = fmul float 0x3FC99999A0000000, %12
  %14 = load i32, ptr %5, align 4
  %15 = sext i32 %14 to i64
  %16 = getelementptr [10 x float], ptr %3, i64 0, i64 %15
  store float %13, ptr %16, align 4
  %17 = load i32, ptr %5, align 4
  %18 = sitofp i32 %17 to float
  %19 = fmul float 0x3FCC28F5C0000000, %18
  %20 = load i32, ptr %5, align 4
  %21 = sext i32 %20 to i64
  %22 = getelementptr [10 x float], ptr %4, i64 0, i64 %21
  store float %19, ptr %22, align 4
  br label %23

23:                                               ; preds = %10
  %24 = load i32, ptr %5, align 4
  %25 = add i32 %24, 1
  store i32 %25, ptr %5, align 4
  br label %7, !llvm.loop !5

26:                                               ; preds = %7
  store float 1.200000e+01, ptr %6, align 4
  %27 = call noundef ptr @_Z1fv()
  ret i32 0
}

attributes #0 = { mustprogress noinline optnone sspstrong uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="4" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { nobuiltin allocsize(0) "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="4" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #2 = { mustprogress noinline norecurse optnone sspstrong uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="4" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #3 = { builtin allocsize(0) }

!llvm.module.flags = !{!0, !1, !2, !3}
!llvm.ident = !{!4}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"uwtable", i32 2}
!3 = !{i32 7, !"frame-pointer", i32 2}
!4 = !{!"clang version 17.0.6"}
!5 = distinct !{!5, !6}
!6 = !{!"llvm.loop.mustprogress"}

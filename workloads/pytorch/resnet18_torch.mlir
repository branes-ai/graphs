"builtin.module"() ({
  "func.func"() <{function_type = (!torch.vtensor<[64,3,7,7],f32>, !torch.vtensor<[64],f32>, !torch.vtensor<[64],f32>, !torch.vtensor<[64,64,3,3],f32>, !torch.vtensor<[64],f32>, !torch.vtensor<[64],f32>, !torch.vtensor<[64,64,3,3],f32>, !torch.vtensor<[64],f32>, !torch.vtensor<[64],f32>, !torch.vtensor<[64,64,3,3],f32>, !torch.vtensor<[64],f32>, !torch.vtensor<[64],f32>, !torch.vtensor<[64,64,3,3],f32>, !torch.vtensor<[64],f32>, !torch.vtensor<[64],f32>, !torch.vtensor<[128,64,3,3],f32>, !torch.vtensor<[128],f32>, !torch.vtensor<[128],f32>, !torch.vtensor<[128,128,3,3],f32>, !torch.vtensor<[128],f32>, !torch.vtensor<[128],f32>, !torch.vtensor<[128,64,1,1],f32>, !torch.vtensor<[128],f32>, !torch.vtensor<[128],f32>, !torch.vtensor<[128,128,3,3],f32>, !torch.vtensor<[128],f32>, !torch.vtensor<[128],f32>, !torch.vtensor<[128,128,3,3],f32>, !torch.vtensor<[128],f32>, !torch.vtensor<[128],f32>, !torch.vtensor<[256,128,3,3],f32>, !torch.vtensor<[256],f32>, !torch.vtensor<[256],f32>, !torch.vtensor<[256,256,3,3],f32>, !torch.vtensor<[256],f32>, !torch.vtensor<[256],f32>, !torch.vtensor<[256,128,1,1],f32>, !torch.vtensor<[256],f32>, !torch.vtensor<[256],f32>, !torch.vtensor<[256,256,3,3],f32>, !torch.vtensor<[256],f32>, !torch.vtensor<[256],f32>, !torch.vtensor<[256,256,3,3],f32>, !torch.vtensor<[256],f32>, !torch.vtensor<[256],f32>, !torch.vtensor<[512,256,3,3],f32>, !torch.vtensor<[512],f32>, !torch.vtensor<[512],f32>, !torch.vtensor<[512,512,3,3],f32>, !torch.vtensor<[512],f32>, !torch.vtensor<[512],f32>, !torch.vtensor<[512,256,1,1],f32>, !torch.vtensor<[512],f32>, !torch.vtensor<[512],f32>, !torch.vtensor<[512,512,3,3],f32>, !torch.vtensor<[512],f32>, !torch.vtensor<[512],f32>, !torch.vtensor<[512,512,3,3],f32>, !torch.vtensor<[512],f32>, !torch.vtensor<[512],f32>, !torch.vtensor<[1000,512],f32>, !torch.vtensor<[1000],f32>, !torch.vtensor<[64],f32>, !torch.vtensor<[64],f32>, !torch.vtensor<[],si64>, !torch.vtensor<[64],f32>, !torch.vtensor<[64],f32>, !torch.vtensor<[],si64>, !torch.vtensor<[64],f32>, !torch.vtensor<[64],f32>, !torch.vtensor<[],si64>, !torch.vtensor<[64],f32>, !torch.vtensor<[64],f32>, !torch.vtensor<[],si64>, !torch.vtensor<[64],f32>, !torch.vtensor<[64],f32>, !torch.vtensor<[],si64>, !torch.vtensor<[128],f32>, !torch.vtensor<[128],f32>, !torch.vtensor<[],si64>, !torch.vtensor<[128],f32>, !torch.vtensor<[128],f32>, !torch.vtensor<[],si64>, !torch.vtensor<[128],f32>, !torch.vtensor<[128],f32>, !torch.vtensor<[],si64>, !torch.vtensor<[128],f32>, !torch.vtensor<[128],f32>, !torch.vtensor<[],si64>, !torch.vtensor<[128],f32>, !torch.vtensor<[128],f32>, !torch.vtensor<[],si64>, !torch.vtensor<[256],f32>, !torch.vtensor<[256],f32>, !torch.vtensor<[],si64>, !torch.vtensor<[256],f32>, !torch.vtensor<[256],f32>, !torch.vtensor<[],si64>, !torch.vtensor<[256],f32>, !torch.vtensor<[256],f32>, !torch.vtensor<[],si64>, !torch.vtensor<[256],f32>, !torch.vtensor<[256],f32>, !torch.vtensor<[],si64>, !torch.vtensor<[256],f32>, !torch.vtensor<[256],f32>, !torch.vtensor<[],si64>, !torch.vtensor<[512],f32>, !torch.vtensor<[512],f32>, !torch.vtensor<[],si64>, !torch.vtensor<[512],f32>, !torch.vtensor<[512],f32>, !torch.vtensor<[],si64>, !torch.vtensor<[512],f32>, !torch.vtensor<[512],f32>, !torch.vtensor<[],si64>, !torch.vtensor<[512],f32>, !torch.vtensor<[512],f32>, !torch.vtensor<[],si64>, !torch.vtensor<[512],f32>, !torch.vtensor<[512],f32>, !torch.vtensor<[],si64>, !torch.vtensor<[1,3,224,224],f32>) -> !torch.vtensor<[1,1000],f32>, sym_name = "main"}> ({
  ^bb0(%arg0: !torch.vtensor<[64,3,7,7],f32>, %arg1: !torch.vtensor<[64],f32>, %arg2: !torch.vtensor<[64],f32>, %arg3: !torch.vtensor<[64,64,3,3],f32>, %arg4: !torch.vtensor<[64],f32>, %arg5: !torch.vtensor<[64],f32>, %arg6: !torch.vtensor<[64,64,3,3],f32>, %arg7: !torch.vtensor<[64],f32>, %arg8: !torch.vtensor<[64],f32>, %arg9: !torch.vtensor<[64,64,3,3],f32>, %arg10: !torch.vtensor<[64],f32>, %arg11: !torch.vtensor<[64],f32>, %arg12: !torch.vtensor<[64,64,3,3],f32>, %arg13: !torch.vtensor<[64],f32>, %arg14: !torch.vtensor<[64],f32>, %arg15: !torch.vtensor<[128,64,3,3],f32>, %arg16: !torch.vtensor<[128],f32>, %arg17: !torch.vtensor<[128],f32>, %arg18: !torch.vtensor<[128,128,3,3],f32>, %arg19: !torch.vtensor<[128],f32>, %arg20: !torch.vtensor<[128],f32>, %arg21: !torch.vtensor<[128,64,1,1],f32>, %arg22: !torch.vtensor<[128],f32>, %arg23: !torch.vtensor<[128],f32>, %arg24: !torch.vtensor<[128,128,3,3],f32>, %arg25: !torch.vtensor<[128],f32>, %arg26: !torch.vtensor<[128],f32>, %arg27: !torch.vtensor<[128,128,3,3],f32>, %arg28: !torch.vtensor<[128],f32>, %arg29: !torch.vtensor<[128],f32>, %arg30: !torch.vtensor<[256,128,3,3],f32>, %arg31: !torch.vtensor<[256],f32>, %arg32: !torch.vtensor<[256],f32>, %arg33: !torch.vtensor<[256,256,3,3],f32>, %arg34: !torch.vtensor<[256],f32>, %arg35: !torch.vtensor<[256],f32>, %arg36: !torch.vtensor<[256,128,1,1],f32>, %arg37: !torch.vtensor<[256],f32>, %arg38: !torch.vtensor<[256],f32>, %arg39: !torch.vtensor<[256,256,3,3],f32>, %arg40: !torch.vtensor<[256],f32>, %arg41: !torch.vtensor<[256],f32>, %arg42: !torch.vtensor<[256,256,3,3],f32>, %arg43: !torch.vtensor<[256],f32>, %arg44: !torch.vtensor<[256],f32>, %arg45: !torch.vtensor<[512,256,3,3],f32>, %arg46: !torch.vtensor<[512],f32>, %arg47: !torch.vtensor<[512],f32>, %arg48: !torch.vtensor<[512,512,3,3],f32>, %arg49: !torch.vtensor<[512],f32>, %arg50: !torch.vtensor<[512],f32>, %arg51: !torch.vtensor<[512,256,1,1],f32>, %arg52: !torch.vtensor<[512],f32>, %arg53: !torch.vtensor<[512],f32>, %arg54: !torch.vtensor<[512,512,3,3],f32>, %arg55: !torch.vtensor<[512],f32>, %arg56: !torch.vtensor<[512],f32>, %arg57: !torch.vtensor<[512,512,3,3],f32>, %arg58: !torch.vtensor<[512],f32>, %arg59: !torch.vtensor<[512],f32>, %arg60: !torch.vtensor<[1000,512],f32>, %arg61: !torch.vtensor<[1000],f32>, %arg62: !torch.vtensor<[64],f32>, %arg63: !torch.vtensor<[64],f32>, %arg64: !torch.vtensor<[],si64>, %arg65: !torch.vtensor<[64],f32>, %arg66: !torch.vtensor<[64],f32>, %arg67: !torch.vtensor<[],si64>, %arg68: !torch.vtensor<[64],f32>, %arg69: !torch.vtensor<[64],f32>, %arg70: !torch.vtensor<[],si64>, %arg71: !torch.vtensor<[64],f32>, %arg72: !torch.vtensor<[64],f32>, %arg73: !torch.vtensor<[],si64>, %arg74: !torch.vtensor<[64],f32>, %arg75: !torch.vtensor<[64],f32>, %arg76: !torch.vtensor<[],si64>, %arg77: !torch.vtensor<[128],f32>, %arg78: !torch.vtensor<[128],f32>, %arg79: !torch.vtensor<[],si64>, %arg80: !torch.vtensor<[128],f32>, %arg81: !torch.vtensor<[128],f32>, %arg82: !torch.vtensor<[],si64>, %arg83: !torch.vtensor<[128],f32>, %arg84: !torch.vtensor<[128],f32>, %arg85: !torch.vtensor<[],si64>, %arg86: !torch.vtensor<[128],f32>, %arg87: !torch.vtensor<[128],f32>, %arg88: !torch.vtensor<[],si64>, %arg89: !torch.vtensor<[128],f32>, %arg90: !torch.vtensor<[128],f32>, %arg91: !torch.vtensor<[],si64>, %arg92: !torch.vtensor<[256],f32>, %arg93: !torch.vtensor<[256],f32>, %arg94: !torch.vtensor<[],si64>, %arg95: !torch.vtensor<[256],f32>, %arg96: !torch.vtensor<[256],f32>, %arg97: !torch.vtensor<[],si64>, %arg98: !torch.vtensor<[256],f32>, %arg99: !torch.vtensor<[256],f32>, %arg100: !torch.vtensor<[],si64>, %arg101: !torch.vtensor<[256],f32>, %arg102: !torch.vtensor<[256],f32>, %arg103: !torch.vtensor<[],si64>, %arg104: !torch.vtensor<[256],f32>, %arg105: !torch.vtensor<[256],f32>, %arg106: !torch.vtensor<[],si64>, %arg107: !torch.vtensor<[512],f32>, %arg108: !torch.vtensor<[512],f32>, %arg109: !torch.vtensor<[],si64>, %arg110: !torch.vtensor<[512],f32>, %arg111: !torch.vtensor<[512],f32>, %arg112: !torch.vtensor<[],si64>, %arg113: !torch.vtensor<[512],f32>, %arg114: !torch.vtensor<[512],f32>, %arg115: !torch.vtensor<[],si64>, %arg116: !torch.vtensor<[512],f32>, %arg117: !torch.vtensor<[512],f32>, %arg118: !torch.vtensor<[],si64>, %arg119: !torch.vtensor<[512],f32>, %arg120: !torch.vtensor<[512],f32>, %arg121: !torch.vtensor<[],si64>, %arg122: !torch.vtensor<[1,3,224,224],f32>):
    %0 = "torch.constant.none"() : () -> !torch.none
    %1 = "torch.constant.int"() <{value = 2 : i64}> : () -> !torch.int
    %2 = "torch.constant.int"() <{value = 2 : i64}> : () -> !torch.int
    %3 = "torch.prim.ListConstruct"(%1, %2) : (!torch.int, !torch.int) -> !torch.list<int>
    %4 = "torch.constant.int"() <{value = 3 : i64}> : () -> !torch.int
    %5 = "torch.constant.int"() <{value = 3 : i64}> : () -> !torch.int
    %6 = "torch.prim.ListConstruct"(%4, %5) : (!torch.int, !torch.int) -> !torch.list<int>
    %7 = "torch.constant.int"() <{value = 1 : i64}> : () -> !torch.int
    %8 = "torch.constant.int"() <{value = 1 : i64}> : () -> !torch.int
    %9 = "torch.prim.ListConstruct"(%7, %8) : (!torch.int, !torch.int) -> !torch.list<int>
    %10 = "torch.constant.int"() <{value = 1 : i64}> : () -> !torch.int
    %11 = "torch.aten.conv2d"(%arg122, %arg0, %0, %3, %6, %9, %10) : (!torch.vtensor<[1,3,224,224],f32>, !torch.vtensor<[64,3,7,7],f32>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int) -> !torch.vtensor<[1,64,112,112],f32>
    %12 = "torch.constant.bool"() <{value = false}> : () -> !torch.bool
    %13 = "torch.constant.float"() <{value = 1.000000e-01 : f64}> : () -> !torch.float
    %14 = "torch.constant.float"() <{value = 1.000000e-05 : f64}> : () -> !torch.float
    %15 = "torch.constant.bool"() <{value = true}> : () -> !torch.bool
    %16 = "torch.aten.batch_norm"(%11, %arg1, %arg2, %arg62, %arg63, %12, %13, %14, %15) : (!torch.vtensor<[1,64,112,112],f32>, !torch.vtensor<[64],f32>, !torch.vtensor<[64],f32>, !torch.vtensor<[64],f32>, !torch.vtensor<[64],f32>, !torch.bool, !torch.float, !torch.float, !torch.bool) -> !torch.vtensor<[1,64,112,112],f32>
    %17 = "torch.aten.relu_"(%16) : (!torch.vtensor<[1,64,112,112],f32>) -> !torch.vtensor<[1,64,112,112],f32>
    %18 = "torch.constant.int"() <{value = 3 : i64}> : () -> !torch.int
    %19 = "torch.constant.int"() <{value = 3 : i64}> : () -> !torch.int
    %20 = "torch.prim.ListConstruct"(%18, %19) : (!torch.int, !torch.int) -> !torch.list<int>
    %21 = "torch.constant.int"() <{value = 2 : i64}> : () -> !torch.int
    %22 = "torch.constant.int"() <{value = 2 : i64}> : () -> !torch.int
    %23 = "torch.prim.ListConstruct"(%21, %22) : (!torch.int, !torch.int) -> !torch.list<int>
    %24 = "torch.constant.int"() <{value = 1 : i64}> : () -> !torch.int
    %25 = "torch.constant.int"() <{value = 1 : i64}> : () -> !torch.int
    %26 = "torch.prim.ListConstruct"(%24, %25) : (!torch.int, !torch.int) -> !torch.list<int>
    %27 = "torch.constant.int"() <{value = 1 : i64}> : () -> !torch.int
    %28 = "torch.constant.int"() <{value = 1 : i64}> : () -> !torch.int
    %29 = "torch.prim.ListConstruct"(%27, %28) : (!torch.int, !torch.int) -> !torch.list<int>
    %30 = "torch.constant.bool"() <{value = false}> : () -> !torch.bool
    %31 = "torch.aten.max_pool2d"(%17, %20, %23, %26, %29, %30) : (!torch.vtensor<[1,64,112,112],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool) -> !torch.vtensor<[1,64,56,56],f32>
    %32 = "torch.constant.none"() : () -> !torch.none
    %33 = "torch.constant.int"() <{value = 1 : i64}> : () -> !torch.int
    %34 = "torch.constant.int"() <{value = 1 : i64}> : () -> !torch.int
    %35 = "torch.prim.ListConstruct"(%33, %34) : (!torch.int, !torch.int) -> !torch.list<int>
    %36 = "torch.constant.int"() <{value = 1 : i64}> : () -> !torch.int
    %37 = "torch.constant.int"() <{value = 1 : i64}> : () -> !torch.int
    %38 = "torch.prim.ListConstruct"(%36, %37) : (!torch.int, !torch.int) -> !torch.list<int>
    %39 = "torch.constant.int"() <{value = 1 : i64}> : () -> !torch.int
    %40 = "torch.constant.int"() <{value = 1 : i64}> : () -> !torch.int
    %41 = "torch.prim.ListConstruct"(%39, %40) : (!torch.int, !torch.int) -> !torch.list<int>
    %42 = "torch.constant.int"() <{value = 1 : i64}> : () -> !torch.int
    %43 = "torch.aten.conv2d"(%31, %arg3, %32, %35, %38, %41, %42) : (!torch.vtensor<[1,64,56,56],f32>, !torch.vtensor<[64,64,3,3],f32>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int) -> !torch.vtensor<[1,64,56,56],f32>
    %44 = "torch.constant.bool"() <{value = false}> : () -> !torch.bool
    %45 = "torch.constant.float"() <{value = 1.000000e-01 : f64}> : () -> !torch.float
    %46 = "torch.constant.float"() <{value = 1.000000e-05 : f64}> : () -> !torch.float
    %47 = "torch.constant.bool"() <{value = true}> : () -> !torch.bool
    %48 = "torch.aten.batch_norm"(%43, %arg4, %arg5, %arg65, %arg66, %44, %45, %46, %47) : (!torch.vtensor<[1,64,56,56],f32>, !torch.vtensor<[64],f32>, !torch.vtensor<[64],f32>, !torch.vtensor<[64],f32>, !torch.vtensor<[64],f32>, !torch.bool, !torch.float, !torch.float, !torch.bool) -> !torch.vtensor<[1,64,56,56],f32>
    %49 = "torch.aten.relu_"(%48) : (!torch.vtensor<[1,64,56,56],f32>) -> !torch.vtensor<[1,64,56,56],f32>
    %50 = "torch.constant.none"() : () -> !torch.none
    %51 = "torch.constant.int"() <{value = 1 : i64}> : () -> !torch.int
    %52 = "torch.constant.int"() <{value = 1 : i64}> : () -> !torch.int
    %53 = "torch.prim.ListConstruct"(%51, %52) : (!torch.int, !torch.int) -> !torch.list<int>
    %54 = "torch.constant.int"() <{value = 1 : i64}> : () -> !torch.int
    %55 = "torch.constant.int"() <{value = 1 : i64}> : () -> !torch.int
    %56 = "torch.prim.ListConstruct"(%54, %55) : (!torch.int, !torch.int) -> !torch.list<int>
    %57 = "torch.constant.int"() <{value = 1 : i64}> : () -> !torch.int
    %58 = "torch.constant.int"() <{value = 1 : i64}> : () -> !torch.int
    %59 = "torch.prim.ListConstruct"(%57, %58) : (!torch.int, !torch.int) -> !torch.list<int>
    %60 = "torch.constant.int"() <{value = 1 : i64}> : () -> !torch.int
    %61 = "torch.aten.conv2d"(%49, %arg6, %50, %53, %56, %59, %60) : (!torch.vtensor<[1,64,56,56],f32>, !torch.vtensor<[64,64,3,3],f32>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int) -> !torch.vtensor<[1,64,56,56],f32>
    %62 = "torch.constant.bool"() <{value = false}> : () -> !torch.bool
    %63 = "torch.constant.float"() <{value = 1.000000e-01 : f64}> : () -> !torch.float
    %64 = "torch.constant.float"() <{value = 1.000000e-05 : f64}> : () -> !torch.float
    %65 = "torch.constant.bool"() <{value = true}> : () -> !torch.bool
    %66 = "torch.aten.batch_norm"(%61, %arg7, %arg8, %arg68, %arg69, %62, %63, %64, %65) : (!torch.vtensor<[1,64,56,56],f32>, !torch.vtensor<[64],f32>, !torch.vtensor<[64],f32>, !torch.vtensor<[64],f32>, !torch.vtensor<[64],f32>, !torch.bool, !torch.float, !torch.float, !torch.bool) -> !torch.vtensor<[1,64,56,56],f32>
    %67 = "torch.constant.int"() <{value = 1 : i64}> : () -> !torch.int
    %68 = "torch.aten.add_.Tensor"(%66, %31, %67) : (!torch.vtensor<[1,64,56,56],f32>, !torch.vtensor<[1,64,56,56],f32>, !torch.int) -> !torch.vtensor<[1,64,56,56],f32>
    %69 = "torch.aten.relu_"(%68) : (!torch.vtensor<[1,64,56,56],f32>) -> !torch.vtensor<[1,64,56,56],f32>
    %70 = "torch.constant.none"() : () -> !torch.none
    %71 = "torch.constant.int"() <{value = 1 : i64}> : () -> !torch.int
    %72 = "torch.constant.int"() <{value = 1 : i64}> : () -> !torch.int
    %73 = "torch.prim.ListConstruct"(%71, %72) : (!torch.int, !torch.int) -> !torch.list<int>
    %74 = "torch.constant.int"() <{value = 1 : i64}> : () -> !torch.int
    %75 = "torch.constant.int"() <{value = 1 : i64}> : () -> !torch.int
    %76 = "torch.prim.ListConstruct"(%74, %75) : (!torch.int, !torch.int) -> !torch.list<int>
    %77 = "torch.constant.int"() <{value = 1 : i64}> : () -> !torch.int
    %78 = "torch.constant.int"() <{value = 1 : i64}> : () -> !torch.int
    %79 = "torch.prim.ListConstruct"(%77, %78) : (!torch.int, !torch.int) -> !torch.list<int>
    %80 = "torch.constant.int"() <{value = 1 : i64}> : () -> !torch.int
    %81 = "torch.aten.conv2d"(%69, %arg9, %70, %73, %76, %79, %80) : (!torch.vtensor<[1,64,56,56],f32>, !torch.vtensor<[64,64,3,3],f32>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int) -> !torch.vtensor<[1,64,56,56],f32>
    %82 = "torch.constant.bool"() <{value = false}> : () -> !torch.bool
    %83 = "torch.constant.float"() <{value = 1.000000e-01 : f64}> : () -> !torch.float
    %84 = "torch.constant.float"() <{value = 1.000000e-05 : f64}> : () -> !torch.float
    %85 = "torch.constant.bool"() <{value = true}> : () -> !torch.bool
    %86 = "torch.aten.batch_norm"(%81, %arg10, %arg11, %arg71, %arg72, %82, %83, %84, %85) : (!torch.vtensor<[1,64,56,56],f32>, !torch.vtensor<[64],f32>, !torch.vtensor<[64],f32>, !torch.vtensor<[64],f32>, !torch.vtensor<[64],f32>, !torch.bool, !torch.float, !torch.float, !torch.bool) -> !torch.vtensor<[1,64,56,56],f32>
    %87 = "torch.aten.relu_"(%86) : (!torch.vtensor<[1,64,56,56],f32>) -> !torch.vtensor<[1,64,56,56],f32>
    %88 = "torch.constant.none"() : () -> !torch.none
    %89 = "torch.constant.int"() <{value = 1 : i64}> : () -> !torch.int
    %90 = "torch.constant.int"() <{value = 1 : i64}> : () -> !torch.int
    %91 = "torch.prim.ListConstruct"(%89, %90) : (!torch.int, !torch.int) -> !torch.list<int>
    %92 = "torch.constant.int"() <{value = 1 : i64}> : () -> !torch.int
    %93 = "torch.constant.int"() <{value = 1 : i64}> : () -> !torch.int
    %94 = "torch.prim.ListConstruct"(%92, %93) : (!torch.int, !torch.int) -> !torch.list<int>
    %95 = "torch.constant.int"() <{value = 1 : i64}> : () -> !torch.int
    %96 = "torch.constant.int"() <{value = 1 : i64}> : () -> !torch.int
    %97 = "torch.prim.ListConstruct"(%95, %96) : (!torch.int, !torch.int) -> !torch.list<int>
    %98 = "torch.constant.int"() <{value = 1 : i64}> : () -> !torch.int
    %99 = "torch.aten.conv2d"(%87, %arg12, %88, %91, %94, %97, %98) : (!torch.vtensor<[1,64,56,56],f32>, !torch.vtensor<[64,64,3,3],f32>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int) -> !torch.vtensor<[1,64,56,56],f32>
    %100 = "torch.constant.bool"() <{value = false}> : () -> !torch.bool
    %101 = "torch.constant.float"() <{value = 1.000000e-01 : f64}> : () -> !torch.float
    %102 = "torch.constant.float"() <{value = 1.000000e-05 : f64}> : () -> !torch.float
    %103 = "torch.constant.bool"() <{value = true}> : () -> !torch.bool
    %104 = "torch.aten.batch_norm"(%99, %arg13, %arg14, %arg74, %arg75, %100, %101, %102, %103) : (!torch.vtensor<[1,64,56,56],f32>, !torch.vtensor<[64],f32>, !torch.vtensor<[64],f32>, !torch.vtensor<[64],f32>, !torch.vtensor<[64],f32>, !torch.bool, !torch.float, !torch.float, !torch.bool) -> !torch.vtensor<[1,64,56,56],f32>
    %105 = "torch.constant.int"() <{value = 1 : i64}> : () -> !torch.int
    %106 = "torch.aten.add_.Tensor"(%104, %69, %105) : (!torch.vtensor<[1,64,56,56],f32>, !torch.vtensor<[1,64,56,56],f32>, !torch.int) -> !torch.vtensor<[1,64,56,56],f32>
    %107 = "torch.aten.relu_"(%106) : (!torch.vtensor<[1,64,56,56],f32>) -> !torch.vtensor<[1,64,56,56],f32>
    %108 = "torch.constant.none"() : () -> !torch.none
    %109 = "torch.constant.int"() <{value = 2 : i64}> : () -> !torch.int
    %110 = "torch.constant.int"() <{value = 2 : i64}> : () -> !torch.int
    %111 = "torch.prim.ListConstruct"(%109, %110) : (!torch.int, !torch.int) -> !torch.list<int>
    %112 = "torch.constant.int"() <{value = 1 : i64}> : () -> !torch.int
    %113 = "torch.constant.int"() <{value = 1 : i64}> : () -> !torch.int
    %114 = "torch.prim.ListConstruct"(%112, %113) : (!torch.int, !torch.int) -> !torch.list<int>
    %115 = "torch.constant.int"() <{value = 1 : i64}> : () -> !torch.int
    %116 = "torch.constant.int"() <{value = 1 : i64}> : () -> !torch.int
    %117 = "torch.prim.ListConstruct"(%115, %116) : (!torch.int, !torch.int) -> !torch.list<int>
    %118 = "torch.constant.int"() <{value = 1 : i64}> : () -> !torch.int
    %119 = "torch.aten.conv2d"(%107, %arg15, %108, %111, %114, %117, %118) : (!torch.vtensor<[1,64,56,56],f32>, !torch.vtensor<[128,64,3,3],f32>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int) -> !torch.vtensor<[1,128,28,28],f32>
    %120 = "torch.constant.bool"() <{value = false}> : () -> !torch.bool
    %121 = "torch.constant.float"() <{value = 1.000000e-01 : f64}> : () -> !torch.float
    %122 = "torch.constant.float"() <{value = 1.000000e-05 : f64}> : () -> !torch.float
    %123 = "torch.constant.bool"() <{value = true}> : () -> !torch.bool
    %124 = "torch.aten.batch_norm"(%119, %arg16, %arg17, %arg77, %arg78, %120, %121, %122, %123) : (!torch.vtensor<[1,128,28,28],f32>, !torch.vtensor<[128],f32>, !torch.vtensor<[128],f32>, !torch.vtensor<[128],f32>, !torch.vtensor<[128],f32>, !torch.bool, !torch.float, !torch.float, !torch.bool) -> !torch.vtensor<[1,128,28,28],f32>
    %125 = "torch.aten.relu_"(%124) : (!torch.vtensor<[1,128,28,28],f32>) -> !torch.vtensor<[1,128,28,28],f32>
    %126 = "torch.constant.none"() : () -> !torch.none
    %127 = "torch.constant.int"() <{value = 1 : i64}> : () -> !torch.int
    %128 = "torch.constant.int"() <{value = 1 : i64}> : () -> !torch.int
    %129 = "torch.prim.ListConstruct"(%127, %128) : (!torch.int, !torch.int) -> !torch.list<int>
    %130 = "torch.constant.int"() <{value = 1 : i64}> : () -> !torch.int
    %131 = "torch.constant.int"() <{value = 1 : i64}> : () -> !torch.int
    %132 = "torch.prim.ListConstruct"(%130, %131) : (!torch.int, !torch.int) -> !torch.list<int>
    %133 = "torch.constant.int"() <{value = 1 : i64}> : () -> !torch.int
    %134 = "torch.constant.int"() <{value = 1 : i64}> : () -> !torch.int
    %135 = "torch.prim.ListConstruct"(%133, %134) : (!torch.int, !torch.int) -> !torch.list<int>
    %136 = "torch.constant.int"() <{value = 1 : i64}> : () -> !torch.int
    %137 = "torch.aten.conv2d"(%125, %arg18, %126, %129, %132, %135, %136) : (!torch.vtensor<[1,128,28,28],f32>, !torch.vtensor<[128,128,3,3],f32>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int) -> !torch.vtensor<[1,128,28,28],f32>
    %138 = "torch.constant.bool"() <{value = false}> : () -> !torch.bool
    %139 = "torch.constant.float"() <{value = 1.000000e-01 : f64}> : () -> !torch.float
    %140 = "torch.constant.float"() <{value = 1.000000e-05 : f64}> : () -> !torch.float
    %141 = "torch.constant.bool"() <{value = true}> : () -> !torch.bool
    %142 = "torch.aten.batch_norm"(%137, %arg19, %arg20, %arg80, %arg81, %138, %139, %140, %141) : (!torch.vtensor<[1,128,28,28],f32>, !torch.vtensor<[128],f32>, !torch.vtensor<[128],f32>, !torch.vtensor<[128],f32>, !torch.vtensor<[128],f32>, !torch.bool, !torch.float, !torch.float, !torch.bool) -> !torch.vtensor<[1,128,28,28],f32>
    %143 = "torch.constant.none"() : () -> !torch.none
    %144 = "torch.constant.int"() <{value = 2 : i64}> : () -> !torch.int
    %145 = "torch.constant.int"() <{value = 2 : i64}> : () -> !torch.int
    %146 = "torch.prim.ListConstruct"(%144, %145) : (!torch.int, !torch.int) -> !torch.list<int>
    %147 = "torch.constant.int"() <{value = 0 : i64}> : () -> !torch.int
    %148 = "torch.constant.int"() <{value = 0 : i64}> : () -> !torch.int
    %149 = "torch.prim.ListConstruct"(%147, %148) : (!torch.int, !torch.int) -> !torch.list<int>
    %150 = "torch.constant.int"() <{value = 1 : i64}> : () -> !torch.int
    %151 = "torch.constant.int"() <{value = 1 : i64}> : () -> !torch.int
    %152 = "torch.prim.ListConstruct"(%150, %151) : (!torch.int, !torch.int) -> !torch.list<int>
    %153 = "torch.constant.int"() <{value = 1 : i64}> : () -> !torch.int
    %154 = "torch.aten.conv2d"(%107, %arg21, %143, %146, %149, %152, %153) : (!torch.vtensor<[1,64,56,56],f32>, !torch.vtensor<[128,64,1,1],f32>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int) -> !torch.vtensor<[1,128,28,28],f32>
    %155 = "torch.constant.bool"() <{value = false}> : () -> !torch.bool
    %156 = "torch.constant.float"() <{value = 1.000000e-01 : f64}> : () -> !torch.float
    %157 = "torch.constant.float"() <{value = 1.000000e-05 : f64}> : () -> !torch.float
    %158 = "torch.constant.bool"() <{value = true}> : () -> !torch.bool
    %159 = "torch.aten.batch_norm"(%154, %arg22, %arg23, %arg83, %arg84, %155, %156, %157, %158) : (!torch.vtensor<[1,128,28,28],f32>, !torch.vtensor<[128],f32>, !torch.vtensor<[128],f32>, !torch.vtensor<[128],f32>, !torch.vtensor<[128],f32>, !torch.bool, !torch.float, !torch.float, !torch.bool) -> !torch.vtensor<[1,128,28,28],f32>
    %160 = "torch.constant.int"() <{value = 1 : i64}> : () -> !torch.int
    %161 = "torch.aten.add_.Tensor"(%142, %159, %160) : (!torch.vtensor<[1,128,28,28],f32>, !torch.vtensor<[1,128,28,28],f32>, !torch.int) -> !torch.vtensor<[1,128,28,28],f32>
    %162 = "torch.aten.relu_"(%161) : (!torch.vtensor<[1,128,28,28],f32>) -> !torch.vtensor<[1,128,28,28],f32>
    %163 = "torch.constant.none"() : () -> !torch.none
    %164 = "torch.constant.int"() <{value = 1 : i64}> : () -> !torch.int
    %165 = "torch.constant.int"() <{value = 1 : i64}> : () -> !torch.int
    %166 = "torch.prim.ListConstruct"(%164, %165) : (!torch.int, !torch.int) -> !torch.list<int>
    %167 = "torch.constant.int"() <{value = 1 : i64}> : () -> !torch.int
    %168 = "torch.constant.int"() <{value = 1 : i64}> : () -> !torch.int
    %169 = "torch.prim.ListConstruct"(%167, %168) : (!torch.int, !torch.int) -> !torch.list<int>
    %170 = "torch.constant.int"() <{value = 1 : i64}> : () -> !torch.int
    %171 = "torch.constant.int"() <{value = 1 : i64}> : () -> !torch.int
    %172 = "torch.prim.ListConstruct"(%170, %171) : (!torch.int, !torch.int) -> !torch.list<int>
    %173 = "torch.constant.int"() <{value = 1 : i64}> : () -> !torch.int
    %174 = "torch.aten.conv2d"(%162, %arg24, %163, %166, %169, %172, %173) : (!torch.vtensor<[1,128,28,28],f32>, !torch.vtensor<[128,128,3,3],f32>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int) -> !torch.vtensor<[1,128,28,28],f32>
    %175 = "torch.constant.bool"() <{value = false}> : () -> !torch.bool
    %176 = "torch.constant.float"() <{value = 1.000000e-01 : f64}> : () -> !torch.float
    %177 = "torch.constant.float"() <{value = 1.000000e-05 : f64}> : () -> !torch.float
    %178 = "torch.constant.bool"() <{value = true}> : () -> !torch.bool
    %179 = "torch.aten.batch_norm"(%174, %arg25, %arg26, %arg86, %arg87, %175, %176, %177, %178) : (!torch.vtensor<[1,128,28,28],f32>, !torch.vtensor<[128],f32>, !torch.vtensor<[128],f32>, !torch.vtensor<[128],f32>, !torch.vtensor<[128],f32>, !torch.bool, !torch.float, !torch.float, !torch.bool) -> !torch.vtensor<[1,128,28,28],f32>
    %180 = "torch.aten.relu_"(%179) : (!torch.vtensor<[1,128,28,28],f32>) -> !torch.vtensor<[1,128,28,28],f32>
    %181 = "torch.constant.none"() : () -> !torch.none
    %182 = "torch.constant.int"() <{value = 1 : i64}> : () -> !torch.int
    %183 = "torch.constant.int"() <{value = 1 : i64}> : () -> !torch.int
    %184 = "torch.prim.ListConstruct"(%182, %183) : (!torch.int, !torch.int) -> !torch.list<int>
    %185 = "torch.constant.int"() <{value = 1 : i64}> : () -> !torch.int
    %186 = "torch.constant.int"() <{value = 1 : i64}> : () -> !torch.int
    %187 = "torch.prim.ListConstruct"(%185, %186) : (!torch.int, !torch.int) -> !torch.list<int>
    %188 = "torch.constant.int"() <{value = 1 : i64}> : () -> !torch.int
    %189 = "torch.constant.int"() <{value = 1 : i64}> : () -> !torch.int
    %190 = "torch.prim.ListConstruct"(%188, %189) : (!torch.int, !torch.int) -> !torch.list<int>
    %191 = "torch.constant.int"() <{value = 1 : i64}> : () -> !torch.int
    %192 = "torch.aten.conv2d"(%180, %arg27, %181, %184, %187, %190, %191) : (!torch.vtensor<[1,128,28,28],f32>, !torch.vtensor<[128,128,3,3],f32>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int) -> !torch.vtensor<[1,128,28,28],f32>
    %193 = "torch.constant.bool"() <{value = false}> : () -> !torch.bool
    %194 = "torch.constant.float"() <{value = 1.000000e-01 : f64}> : () -> !torch.float
    %195 = "torch.constant.float"() <{value = 1.000000e-05 : f64}> : () -> !torch.float
    %196 = "torch.constant.bool"() <{value = true}> : () -> !torch.bool
    %197 = "torch.aten.batch_norm"(%192, %arg28, %arg29, %arg89, %arg90, %193, %194, %195, %196) : (!torch.vtensor<[1,128,28,28],f32>, !torch.vtensor<[128],f32>, !torch.vtensor<[128],f32>, !torch.vtensor<[128],f32>, !torch.vtensor<[128],f32>, !torch.bool, !torch.float, !torch.float, !torch.bool) -> !torch.vtensor<[1,128,28,28],f32>
    %198 = "torch.constant.int"() <{value = 1 : i64}> : () -> !torch.int
    %199 = "torch.aten.add_.Tensor"(%197, %162, %198) : (!torch.vtensor<[1,128,28,28],f32>, !torch.vtensor<[1,128,28,28],f32>, !torch.int) -> !torch.vtensor<[1,128,28,28],f32>
    %200 = "torch.aten.relu_"(%199) : (!torch.vtensor<[1,128,28,28],f32>) -> !torch.vtensor<[1,128,28,28],f32>
    %201 = "torch.constant.none"() : () -> !torch.none
    %202 = "torch.constant.int"() <{value = 2 : i64}> : () -> !torch.int
    %203 = "torch.constant.int"() <{value = 2 : i64}> : () -> !torch.int
    %204 = "torch.prim.ListConstruct"(%202, %203) : (!torch.int, !torch.int) -> !torch.list<int>
    %205 = "torch.constant.int"() <{value = 1 : i64}> : () -> !torch.int
    %206 = "torch.constant.int"() <{value = 1 : i64}> : () -> !torch.int
    %207 = "torch.prim.ListConstruct"(%205, %206) : (!torch.int, !torch.int) -> !torch.list<int>
    %208 = "torch.constant.int"() <{value = 1 : i64}> : () -> !torch.int
    %209 = "torch.constant.int"() <{value = 1 : i64}> : () -> !torch.int
    %210 = "torch.prim.ListConstruct"(%208, %209) : (!torch.int, !torch.int) -> !torch.list<int>
    %211 = "torch.constant.int"() <{value = 1 : i64}> : () -> !torch.int
    %212 = "torch.aten.conv2d"(%200, %arg30, %201, %204, %207, %210, %211) : (!torch.vtensor<[1,128,28,28],f32>, !torch.vtensor<[256,128,3,3],f32>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int) -> !torch.vtensor<[1,256,14,14],f32>
    %213 = "torch.constant.bool"() <{value = false}> : () -> !torch.bool
    %214 = "torch.constant.float"() <{value = 1.000000e-01 : f64}> : () -> !torch.float
    %215 = "torch.constant.float"() <{value = 1.000000e-05 : f64}> : () -> !torch.float
    %216 = "torch.constant.bool"() <{value = true}> : () -> !torch.bool
    %217 = "torch.aten.batch_norm"(%212, %arg31, %arg32, %arg92, %arg93, %213, %214, %215, %216) : (!torch.vtensor<[1,256,14,14],f32>, !torch.vtensor<[256],f32>, !torch.vtensor<[256],f32>, !torch.vtensor<[256],f32>, !torch.vtensor<[256],f32>, !torch.bool, !torch.float, !torch.float, !torch.bool) -> !torch.vtensor<[1,256,14,14],f32>
    %218 = "torch.aten.relu_"(%217) : (!torch.vtensor<[1,256,14,14],f32>) -> !torch.vtensor<[1,256,14,14],f32>
    %219 = "torch.constant.none"() : () -> !torch.none
    %220 = "torch.constant.int"() <{value = 1 : i64}> : () -> !torch.int
    %221 = "torch.constant.int"() <{value = 1 : i64}> : () -> !torch.int
    %222 = "torch.prim.ListConstruct"(%220, %221) : (!torch.int, !torch.int) -> !torch.list<int>
    %223 = "torch.constant.int"() <{value = 1 : i64}> : () -> !torch.int
    %224 = "torch.constant.int"() <{value = 1 : i64}> : () -> !torch.int
    %225 = "torch.prim.ListConstruct"(%223, %224) : (!torch.int, !torch.int) -> !torch.list<int>
    %226 = "torch.constant.int"() <{value = 1 : i64}> : () -> !torch.int
    %227 = "torch.constant.int"() <{value = 1 : i64}> : () -> !torch.int
    %228 = "torch.prim.ListConstruct"(%226, %227) : (!torch.int, !torch.int) -> !torch.list<int>
    %229 = "torch.constant.int"() <{value = 1 : i64}> : () -> !torch.int
    %230 = "torch.aten.conv2d"(%218, %arg33, %219, %222, %225, %228, %229) : (!torch.vtensor<[1,256,14,14],f32>, !torch.vtensor<[256,256,3,3],f32>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int) -> !torch.vtensor<[1,256,14,14],f32>
    %231 = "torch.constant.bool"() <{value = false}> : () -> !torch.bool
    %232 = "torch.constant.float"() <{value = 1.000000e-01 : f64}> : () -> !torch.float
    %233 = "torch.constant.float"() <{value = 1.000000e-05 : f64}> : () -> !torch.float
    %234 = "torch.constant.bool"() <{value = true}> : () -> !torch.bool
    %235 = "torch.aten.batch_norm"(%230, %arg34, %arg35, %arg95, %arg96, %231, %232, %233, %234) : (!torch.vtensor<[1,256,14,14],f32>, !torch.vtensor<[256],f32>, !torch.vtensor<[256],f32>, !torch.vtensor<[256],f32>, !torch.vtensor<[256],f32>, !torch.bool, !torch.float, !torch.float, !torch.bool) -> !torch.vtensor<[1,256,14,14],f32>
    %236 = "torch.constant.none"() : () -> !torch.none
    %237 = "torch.constant.int"() <{value = 2 : i64}> : () -> !torch.int
    %238 = "torch.constant.int"() <{value = 2 : i64}> : () -> !torch.int
    %239 = "torch.prim.ListConstruct"(%237, %238) : (!torch.int, !torch.int) -> !torch.list<int>
    %240 = "torch.constant.int"() <{value = 0 : i64}> : () -> !torch.int
    %241 = "torch.constant.int"() <{value = 0 : i64}> : () -> !torch.int
    %242 = "torch.prim.ListConstruct"(%240, %241) : (!torch.int, !torch.int) -> !torch.list<int>
    %243 = "torch.constant.int"() <{value = 1 : i64}> : () -> !torch.int
    %244 = "torch.constant.int"() <{value = 1 : i64}> : () -> !torch.int
    %245 = "torch.prim.ListConstruct"(%243, %244) : (!torch.int, !torch.int) -> !torch.list<int>
    %246 = "torch.constant.int"() <{value = 1 : i64}> : () -> !torch.int
    %247 = "torch.aten.conv2d"(%200, %arg36, %236, %239, %242, %245, %246) : (!torch.vtensor<[1,128,28,28],f32>, !torch.vtensor<[256,128,1,1],f32>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int) -> !torch.vtensor<[1,256,14,14],f32>
    %248 = "torch.constant.bool"() <{value = false}> : () -> !torch.bool
    %249 = "torch.constant.float"() <{value = 1.000000e-01 : f64}> : () -> !torch.float
    %250 = "torch.constant.float"() <{value = 1.000000e-05 : f64}> : () -> !torch.float
    %251 = "torch.constant.bool"() <{value = true}> : () -> !torch.bool
    %252 = "torch.aten.batch_norm"(%247, %arg37, %arg38, %arg98, %arg99, %248, %249, %250, %251) : (!torch.vtensor<[1,256,14,14],f32>, !torch.vtensor<[256],f32>, !torch.vtensor<[256],f32>, !torch.vtensor<[256],f32>, !torch.vtensor<[256],f32>, !torch.bool, !torch.float, !torch.float, !torch.bool) -> !torch.vtensor<[1,256,14,14],f32>
    %253 = "torch.constant.int"() <{value = 1 : i64}> : () -> !torch.int
    %254 = "torch.aten.add_.Tensor"(%235, %252, %253) : (!torch.vtensor<[1,256,14,14],f32>, !torch.vtensor<[1,256,14,14],f32>, !torch.int) -> !torch.vtensor<[1,256,14,14],f32>
    %255 = "torch.aten.relu_"(%254) : (!torch.vtensor<[1,256,14,14],f32>) -> !torch.vtensor<[1,256,14,14],f32>
    %256 = "torch.constant.none"() : () -> !torch.none
    %257 = "torch.constant.int"() <{value = 1 : i64}> : () -> !torch.int
    %258 = "torch.constant.int"() <{value = 1 : i64}> : () -> !torch.int
    %259 = "torch.prim.ListConstruct"(%257, %258) : (!torch.int, !torch.int) -> !torch.list<int>
    %260 = "torch.constant.int"() <{value = 1 : i64}> : () -> !torch.int
    %261 = "torch.constant.int"() <{value = 1 : i64}> : () -> !torch.int
    %262 = "torch.prim.ListConstruct"(%260, %261) : (!torch.int, !torch.int) -> !torch.list<int>
    %263 = "torch.constant.int"() <{value = 1 : i64}> : () -> !torch.int
    %264 = "torch.constant.int"() <{value = 1 : i64}> : () -> !torch.int
    %265 = "torch.prim.ListConstruct"(%263, %264) : (!torch.int, !torch.int) -> !torch.list<int>
    %266 = "torch.constant.int"() <{value = 1 : i64}> : () -> !torch.int
    %267 = "torch.aten.conv2d"(%255, %arg39, %256, %259, %262, %265, %266) : (!torch.vtensor<[1,256,14,14],f32>, !torch.vtensor<[256,256,3,3],f32>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int) -> !torch.vtensor<[1,256,14,14],f32>
    %268 = "torch.constant.bool"() <{value = false}> : () -> !torch.bool
    %269 = "torch.constant.float"() <{value = 1.000000e-01 : f64}> : () -> !torch.float
    %270 = "torch.constant.float"() <{value = 1.000000e-05 : f64}> : () -> !torch.float
    %271 = "torch.constant.bool"() <{value = true}> : () -> !torch.bool
    %272 = "torch.aten.batch_norm"(%267, %arg40, %arg41, %arg101, %arg102, %268, %269, %270, %271) : (!torch.vtensor<[1,256,14,14],f32>, !torch.vtensor<[256],f32>, !torch.vtensor<[256],f32>, !torch.vtensor<[256],f32>, !torch.vtensor<[256],f32>, !torch.bool, !torch.float, !torch.float, !torch.bool) -> !torch.vtensor<[1,256,14,14],f32>
    %273 = "torch.aten.relu_"(%272) : (!torch.vtensor<[1,256,14,14],f32>) -> !torch.vtensor<[1,256,14,14],f32>
    %274 = "torch.constant.none"() : () -> !torch.none
    %275 = "torch.constant.int"() <{value = 1 : i64}> : () -> !torch.int
    %276 = "torch.constant.int"() <{value = 1 : i64}> : () -> !torch.int
    %277 = "torch.prim.ListConstruct"(%275, %276) : (!torch.int, !torch.int) -> !torch.list<int>
    %278 = "torch.constant.int"() <{value = 1 : i64}> : () -> !torch.int
    %279 = "torch.constant.int"() <{value = 1 : i64}> : () -> !torch.int
    %280 = "torch.prim.ListConstruct"(%278, %279) : (!torch.int, !torch.int) -> !torch.list<int>
    %281 = "torch.constant.int"() <{value = 1 : i64}> : () -> !torch.int
    %282 = "torch.constant.int"() <{value = 1 : i64}> : () -> !torch.int
    %283 = "torch.prim.ListConstruct"(%281, %282) : (!torch.int, !torch.int) -> !torch.list<int>
    %284 = "torch.constant.int"() <{value = 1 : i64}> : () -> !torch.int
    %285 = "torch.aten.conv2d"(%273, %arg42, %274, %277, %280, %283, %284) : (!torch.vtensor<[1,256,14,14],f32>, !torch.vtensor<[256,256,3,3],f32>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int) -> !torch.vtensor<[1,256,14,14],f32>
    %286 = "torch.constant.bool"() <{value = false}> : () -> !torch.bool
    %287 = "torch.constant.float"() <{value = 1.000000e-01 : f64}> : () -> !torch.float
    %288 = "torch.constant.float"() <{value = 1.000000e-05 : f64}> : () -> !torch.float
    %289 = "torch.constant.bool"() <{value = true}> : () -> !torch.bool
    %290 = "torch.aten.batch_norm"(%285, %arg43, %arg44, %arg104, %arg105, %286, %287, %288, %289) : (!torch.vtensor<[1,256,14,14],f32>, !torch.vtensor<[256],f32>, !torch.vtensor<[256],f32>, !torch.vtensor<[256],f32>, !torch.vtensor<[256],f32>, !torch.bool, !torch.float, !torch.float, !torch.bool) -> !torch.vtensor<[1,256,14,14],f32>
    %291 = "torch.constant.int"() <{value = 1 : i64}> : () -> !torch.int
    %292 = "torch.aten.add_.Tensor"(%290, %255, %291) : (!torch.vtensor<[1,256,14,14],f32>, !torch.vtensor<[1,256,14,14],f32>, !torch.int) -> !torch.vtensor<[1,256,14,14],f32>
    %293 = "torch.aten.relu_"(%292) : (!torch.vtensor<[1,256,14,14],f32>) -> !torch.vtensor<[1,256,14,14],f32>
    %294 = "torch.constant.none"() : () -> !torch.none
    %295 = "torch.constant.int"() <{value = 2 : i64}> : () -> !torch.int
    %296 = "torch.constant.int"() <{value = 2 : i64}> : () -> !torch.int
    %297 = "torch.prim.ListConstruct"(%295, %296) : (!torch.int, !torch.int) -> !torch.list<int>
    %298 = "torch.constant.int"() <{value = 1 : i64}> : () -> !torch.int
    %299 = "torch.constant.int"() <{value = 1 : i64}> : () -> !torch.int
    %300 = "torch.prim.ListConstruct"(%298, %299) : (!torch.int, !torch.int) -> !torch.list<int>
    %301 = "torch.constant.int"() <{value = 1 : i64}> : () -> !torch.int
    %302 = "torch.constant.int"() <{value = 1 : i64}> : () -> !torch.int
    %303 = "torch.prim.ListConstruct"(%301, %302) : (!torch.int, !torch.int) -> !torch.list<int>
    %304 = "torch.constant.int"() <{value = 1 : i64}> : () -> !torch.int
    %305 = "torch.aten.conv2d"(%293, %arg45, %294, %297, %300, %303, %304) : (!torch.vtensor<[1,256,14,14],f32>, !torch.vtensor<[512,256,3,3],f32>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int) -> !torch.vtensor<[1,512,7,7],f32>
    %306 = "torch.constant.bool"() <{value = false}> : () -> !torch.bool
    %307 = "torch.constant.float"() <{value = 1.000000e-01 : f64}> : () -> !torch.float
    %308 = "torch.constant.float"() <{value = 1.000000e-05 : f64}> : () -> !torch.float
    %309 = "torch.constant.bool"() <{value = true}> : () -> !torch.bool
    %310 = "torch.aten.batch_norm"(%305, %arg46, %arg47, %arg107, %arg108, %306, %307, %308, %309) : (!torch.vtensor<[1,512,7,7],f32>, !torch.vtensor<[512],f32>, !torch.vtensor<[512],f32>, !torch.vtensor<[512],f32>, !torch.vtensor<[512],f32>, !torch.bool, !torch.float, !torch.float, !torch.bool) -> !torch.vtensor<[1,512,7,7],f32>
    %311 = "torch.aten.relu_"(%310) : (!torch.vtensor<[1,512,7,7],f32>) -> !torch.vtensor<[1,512,7,7],f32>
    %312 = "torch.constant.none"() : () -> !torch.none
    %313 = "torch.constant.int"() <{value = 1 : i64}> : () -> !torch.int
    %314 = "torch.constant.int"() <{value = 1 : i64}> : () -> !torch.int
    %315 = "torch.prim.ListConstruct"(%313, %314) : (!torch.int, !torch.int) -> !torch.list<int>
    %316 = "torch.constant.int"() <{value = 1 : i64}> : () -> !torch.int
    %317 = "torch.constant.int"() <{value = 1 : i64}> : () -> !torch.int
    %318 = "torch.prim.ListConstruct"(%316, %317) : (!torch.int, !torch.int) -> !torch.list<int>
    %319 = "torch.constant.int"() <{value = 1 : i64}> : () -> !torch.int
    %320 = "torch.constant.int"() <{value = 1 : i64}> : () -> !torch.int
    %321 = "torch.prim.ListConstruct"(%319, %320) : (!torch.int, !torch.int) -> !torch.list<int>
    %322 = "torch.constant.int"() <{value = 1 : i64}> : () -> !torch.int
    %323 = "torch.aten.conv2d"(%311, %arg48, %312, %315, %318, %321, %322) : (!torch.vtensor<[1,512,7,7],f32>, !torch.vtensor<[512,512,3,3],f32>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int) -> !torch.vtensor<[1,512,7,7],f32>
    %324 = "torch.constant.bool"() <{value = false}> : () -> !torch.bool
    %325 = "torch.constant.float"() <{value = 1.000000e-01 : f64}> : () -> !torch.float
    %326 = "torch.constant.float"() <{value = 1.000000e-05 : f64}> : () -> !torch.float
    %327 = "torch.constant.bool"() <{value = true}> : () -> !torch.bool
    %328 = "torch.aten.batch_norm"(%323, %arg49, %arg50, %arg110, %arg111, %324, %325, %326, %327) : (!torch.vtensor<[1,512,7,7],f32>, !torch.vtensor<[512],f32>, !torch.vtensor<[512],f32>, !torch.vtensor<[512],f32>, !torch.vtensor<[512],f32>, !torch.bool, !torch.float, !torch.float, !torch.bool) -> !torch.vtensor<[1,512,7,7],f32>
    %329 = "torch.constant.none"() : () -> !torch.none
    %330 = "torch.constant.int"() <{value = 2 : i64}> : () -> !torch.int
    %331 = "torch.constant.int"() <{value = 2 : i64}> : () -> !torch.int
    %332 = "torch.prim.ListConstruct"(%330, %331) : (!torch.int, !torch.int) -> !torch.list<int>
    %333 = "torch.constant.int"() <{value = 0 : i64}> : () -> !torch.int
    %334 = "torch.constant.int"() <{value = 0 : i64}> : () -> !torch.int
    %335 = "torch.prim.ListConstruct"(%333, %334) : (!torch.int, !torch.int) -> !torch.list<int>
    %336 = "torch.constant.int"() <{value = 1 : i64}> : () -> !torch.int
    %337 = "torch.constant.int"() <{value = 1 : i64}> : () -> !torch.int
    %338 = "torch.prim.ListConstruct"(%336, %337) : (!torch.int, !torch.int) -> !torch.list<int>
    %339 = "torch.constant.int"() <{value = 1 : i64}> : () -> !torch.int
    %340 = "torch.aten.conv2d"(%293, %arg51, %329, %332, %335, %338, %339) : (!torch.vtensor<[1,256,14,14],f32>, !torch.vtensor<[512,256,1,1],f32>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int) -> !torch.vtensor<[1,512,7,7],f32>
    %341 = "torch.constant.bool"() <{value = false}> : () -> !torch.bool
    %342 = "torch.constant.float"() <{value = 1.000000e-01 : f64}> : () -> !torch.float
    %343 = "torch.constant.float"() <{value = 1.000000e-05 : f64}> : () -> !torch.float
    %344 = "torch.constant.bool"() <{value = true}> : () -> !torch.bool
    %345 = "torch.aten.batch_norm"(%340, %arg52, %arg53, %arg113, %arg114, %341, %342, %343, %344) : (!torch.vtensor<[1,512,7,7],f32>, !torch.vtensor<[512],f32>, !torch.vtensor<[512],f32>, !torch.vtensor<[512],f32>, !torch.vtensor<[512],f32>, !torch.bool, !torch.float, !torch.float, !torch.bool) -> !torch.vtensor<[1,512,7,7],f32>
    %346 = "torch.constant.int"() <{value = 1 : i64}> : () -> !torch.int
    %347 = "torch.aten.add_.Tensor"(%328, %345, %346) : (!torch.vtensor<[1,512,7,7],f32>, !torch.vtensor<[1,512,7,7],f32>, !torch.int) -> !torch.vtensor<[1,512,7,7],f32>
    %348 = "torch.aten.relu_"(%347) : (!torch.vtensor<[1,512,7,7],f32>) -> !torch.vtensor<[1,512,7,7],f32>
    %349 = "torch.constant.none"() : () -> !torch.none
    %350 = "torch.constant.int"() <{value = 1 : i64}> : () -> !torch.int
    %351 = "torch.constant.int"() <{value = 1 : i64}> : () -> !torch.int
    %352 = "torch.prim.ListConstruct"(%350, %351) : (!torch.int, !torch.int) -> !torch.list<int>
    %353 = "torch.constant.int"() <{value = 1 : i64}> : () -> !torch.int
    %354 = "torch.constant.int"() <{value = 1 : i64}> : () -> !torch.int
    %355 = "torch.prim.ListConstruct"(%353, %354) : (!torch.int, !torch.int) -> !torch.list<int>
    %356 = "torch.constant.int"() <{value = 1 : i64}> : () -> !torch.int
    %357 = "torch.constant.int"() <{value = 1 : i64}> : () -> !torch.int
    %358 = "torch.prim.ListConstruct"(%356, %357) : (!torch.int, !torch.int) -> !torch.list<int>
    %359 = "torch.constant.int"() <{value = 1 : i64}> : () -> !torch.int
    %360 = "torch.aten.conv2d"(%348, %arg54, %349, %352, %355, %358, %359) : (!torch.vtensor<[1,512,7,7],f32>, !torch.vtensor<[512,512,3,3],f32>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int) -> !torch.vtensor<[1,512,7,7],f32>
    %361 = "torch.constant.bool"() <{value = false}> : () -> !torch.bool
    %362 = "torch.constant.float"() <{value = 1.000000e-01 : f64}> : () -> !torch.float
    %363 = "torch.constant.float"() <{value = 1.000000e-05 : f64}> : () -> !torch.float
    %364 = "torch.constant.bool"() <{value = true}> : () -> !torch.bool
    %365 = "torch.aten.batch_norm"(%360, %arg55, %arg56, %arg116, %arg117, %361, %362, %363, %364) : (!torch.vtensor<[1,512,7,7],f32>, !torch.vtensor<[512],f32>, !torch.vtensor<[512],f32>, !torch.vtensor<[512],f32>, !torch.vtensor<[512],f32>, !torch.bool, !torch.float, !torch.float, !torch.bool) -> !torch.vtensor<[1,512,7,7],f32>
    %366 = "torch.aten.relu_"(%365) : (!torch.vtensor<[1,512,7,7],f32>) -> !torch.vtensor<[1,512,7,7],f32>
    %367 = "torch.constant.none"() : () -> !torch.none
    %368 = "torch.constant.int"() <{value = 1 : i64}> : () -> !torch.int
    %369 = "torch.constant.int"() <{value = 1 : i64}> : () -> !torch.int
    %370 = "torch.prim.ListConstruct"(%368, %369) : (!torch.int, !torch.int) -> !torch.list<int>
    %371 = "torch.constant.int"() <{value = 1 : i64}> : () -> !torch.int
    %372 = "torch.constant.int"() <{value = 1 : i64}> : () -> !torch.int
    %373 = "torch.prim.ListConstruct"(%371, %372) : (!torch.int, !torch.int) -> !torch.list<int>
    %374 = "torch.constant.int"() <{value = 1 : i64}> : () -> !torch.int
    %375 = "torch.constant.int"() <{value = 1 : i64}> : () -> !torch.int
    %376 = "torch.prim.ListConstruct"(%374, %375) : (!torch.int, !torch.int) -> !torch.list<int>
    %377 = "torch.constant.int"() <{value = 1 : i64}> : () -> !torch.int
    %378 = "torch.aten.conv2d"(%366, %arg57, %367, %370, %373, %376, %377) : (!torch.vtensor<[1,512,7,7],f32>, !torch.vtensor<[512,512,3,3],f32>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int) -> !torch.vtensor<[1,512,7,7],f32>
    %379 = "torch.constant.bool"() <{value = false}> : () -> !torch.bool
    %380 = "torch.constant.float"() <{value = 1.000000e-01 : f64}> : () -> !torch.float
    %381 = "torch.constant.float"() <{value = 1.000000e-05 : f64}> : () -> !torch.float
    %382 = "torch.constant.bool"() <{value = true}> : () -> !torch.bool
    %383 = "torch.aten.batch_norm"(%378, %arg58, %arg59, %arg119, %arg120, %379, %380, %381, %382) : (!torch.vtensor<[1,512,7,7],f32>, !torch.vtensor<[512],f32>, !torch.vtensor<[512],f32>, !torch.vtensor<[512],f32>, !torch.vtensor<[512],f32>, !torch.bool, !torch.float, !torch.float, !torch.bool) -> !torch.vtensor<[1,512,7,7],f32>
    %384 = "torch.constant.int"() <{value = 1 : i64}> : () -> !torch.int
    %385 = "torch.aten.add_.Tensor"(%383, %348, %384) : (!torch.vtensor<[1,512,7,7],f32>, !torch.vtensor<[1,512,7,7],f32>, !torch.int) -> !torch.vtensor<[1,512,7,7],f32>
    %386 = "torch.aten.relu_"(%385) : (!torch.vtensor<[1,512,7,7],f32>) -> !torch.vtensor<[1,512,7,7],f32>
    %387 = "torch.constant.int"() <{value = 1 : i64}> : () -> !torch.int
    %388 = "torch.constant.int"() <{value = 1 : i64}> : () -> !torch.int
    %389 = "torch.prim.ListConstruct"(%387, %388) : (!torch.int, !torch.int) -> !torch.list<int>
    %390 = "torch.aten.adaptive_avg_pool2d"(%386, %389) : (!torch.vtensor<[1,512,7,7],f32>, !torch.list<int>) -> !torch.vtensor<[1,512,1,1],f32>
    %391 = "torch.constant.int"() <{value = 1 : i64}> : () -> !torch.int
    %392 = "torch.constant.int"() <{value = -1 : i64}> : () -> !torch.int
    %393 = "torch.aten.flatten.using_ints"(%390, %391, %392) : (!torch.vtensor<[1,512,1,1],f32>, !torch.int, !torch.int) -> !torch.vtensor<[1,512],f32>
    %394 = "torch.aten.linear"(%393, %arg60, %arg61) : (!torch.vtensor<[1,512],f32>, !torch.vtensor<[1000,512],f32>, !torch.vtensor<[1000],f32>) -> !torch.vtensor<[1,1000],f32>
    "func.return"(%394) : (!torch.vtensor<[1,1000],f32>) -> ()
  }) : () -> ()
}) : () -> ()

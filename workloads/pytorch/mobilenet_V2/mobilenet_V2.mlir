module {
  func.func @main(%arg0: !torch.vtensor<[1,3,224,224],f32>, %arg1: !torch.vtensor<[32,3,3,3],f32>, %arg2: !torch.vtensor<[32],f32>, %arg3: !torch.vtensor<[32],f32>, %arg4: !torch.vtensor<[32],f32>, %arg5: !torch.vtensor<[32],f32>, %arg6: !torch.vtensor<[32,1,3,3],f32>, %arg7: !torch.vtensor<[32],f32>, %arg8: !torch.vtensor<[32],f32>, %arg9: !torch.vtensor<[32],f32>, %arg10: !torch.vtensor<[32],f32>, %arg11: !torch.vtensor<[16,32,1,1],f32>, %arg12: !torch.vtensor<[16],f32>, %arg13: !torch.vtensor<[16],f32>, %arg14: !torch.vtensor<[16],f32>, %arg15: !torch.vtensor<[16],f32>, %arg16: !torch.vtensor<[96,16,1,1],f32>, %arg17: !torch.vtensor<[96],f32>, %arg18: !torch.vtensor<[96],f32>, %arg19: !torch.vtensor<[96],f32>, %arg20: !torch.vtensor<[96],f32>, %arg21: !torch.vtensor<[96,1,3,3],f32>, %arg22: !torch.vtensor<[96],f32>, %arg23: !torch.vtensor<[96],f32>, %arg24: !torch.vtensor<[96],f32>, %arg25: !torch.vtensor<[96],f32>, %arg26: !torch.vtensor<[24,96,1,1],f32>, %arg27: !torch.vtensor<[24],f32>, %arg28: !torch.vtensor<[24],f32>, %arg29: !torch.vtensor<[24],f32>, %arg30: !torch.vtensor<[24],f32>, %arg31: !torch.vtensor<[144,24,1,1],f32>, %arg32: !torch.vtensor<[144],f32>, %arg33: !torch.vtensor<[144],f32>, %arg34: !torch.vtensor<[144],f32>, %arg35: !torch.vtensor<[144],f32>, %arg36: !torch.vtensor<[144,1,3,3],f32>, %arg37: !torch.vtensor<[144],f32>, %arg38: !torch.vtensor<[144],f32>, %arg39: !torch.vtensor<[144],f32>, %arg40: !torch.vtensor<[144],f32>, %arg41: !torch.vtensor<[24,144,1,1],f32>, %arg42: !torch.vtensor<[24],f32>, %arg43: !torch.vtensor<[24],f32>, %arg44: !torch.vtensor<[24],f32>, %arg45: !torch.vtensor<[24],f32>, %arg46: !torch.vtensor<[144,24,1,1],f32>, %arg47: !torch.vtensor<[144],f32>, %arg48: !torch.vtensor<[144],f32>, %arg49: !torch.vtensor<[144],f32>, %arg50: !torch.vtensor<[144],f32>, %arg51: !torch.vtensor<[144,1,3,3],f32>, %arg52: !torch.vtensor<[144],f32>, %arg53: !torch.vtensor<[144],f32>, %arg54: !torch.vtensor<[144],f32>, %arg55: !torch.vtensor<[144],f32>, %arg56: !torch.vtensor<[32,144,1,1],f32>, %arg57: !torch.vtensor<[32],f32>, %arg58: !torch.vtensor<[32],f32>, %arg59: !torch.vtensor<[32],f32>, %arg60: !torch.vtensor<[32],f32>, %arg61: !torch.vtensor<[192,32,1,1],f32>, %arg62: !torch.vtensor<[192],f32>, %arg63: !torch.vtensor<[192],f32>, %arg64: !torch.vtensor<[192],f32>, %arg65: !torch.vtensor<[192],f32>, %arg66: !torch.vtensor<[192,1,3,3],f32>, %arg67: !torch.vtensor<[192],f32>, %arg68: !torch.vtensor<[192],f32>, %arg69: !torch.vtensor<[192],f32>, %arg70: !torch.vtensor<[192],f32>, %arg71: !torch.vtensor<[32,192,1,1],f32>, %arg72: !torch.vtensor<[32],f32>, %arg73: !torch.vtensor<[32],f32>, %arg74: !torch.vtensor<[32],f32>, %arg75: !torch.vtensor<[32],f32>, %arg76: !torch.vtensor<[192,32,1,1],f32>, %arg77: !torch.vtensor<[192],f32>, %arg78: !torch.vtensor<[192],f32>, %arg79: !torch.vtensor<[192],f32>, %arg80: !torch.vtensor<[192],f32>, %arg81: !torch.vtensor<[192,1,3,3],f32>, %arg82: !torch.vtensor<[192],f32>, %arg83: !torch.vtensor<[192],f32>, %arg84: !torch.vtensor<[192],f32>, %arg85: !torch.vtensor<[192],f32>, %arg86: !torch.vtensor<[32,192,1,1],f32>, %arg87: !torch.vtensor<[32],f32>, %arg88: !torch.vtensor<[32],f32>, %arg89: !torch.vtensor<[32],f32>, %arg90: !torch.vtensor<[32],f32>, %arg91: !torch.vtensor<[192,32,1,1],f32>, %arg92: !torch.vtensor<[192],f32>, %arg93: !torch.vtensor<[192],f32>, %arg94: !torch.vtensor<[192],f32>, %arg95: !torch.vtensor<[192],f32>, %arg96: !torch.vtensor<[192,1,3,3],f32>, %arg97: !torch.vtensor<[192],f32>, %arg98: !torch.vtensor<[192],f32>, %arg99: !torch.vtensor<[192],f32>, %arg100: !torch.vtensor<[192],f32>, %arg101: !torch.vtensor<[64,192,1,1],f32>, %arg102: !torch.vtensor<[64],f32>, %arg103: !torch.vtensor<[64],f32>, %arg104: !torch.vtensor<[64],f32>, %arg105: !torch.vtensor<[64],f32>, %arg106: !torch.vtensor<[384,64,1,1],f32>, %arg107: !torch.vtensor<[384],f32>, %arg108: !torch.vtensor<[384],f32>, %arg109: !torch.vtensor<[384],f32>, %arg110: !torch.vtensor<[384],f32>, %arg111: !torch.vtensor<[384,1,3,3],f32>, %arg112: !torch.vtensor<[384],f32>, %arg113: !torch.vtensor<[384],f32>, %arg114: !torch.vtensor<[384],f32>, %arg115: !torch.vtensor<[384],f32>, %arg116: !torch.vtensor<[64,384,1,1],f32>, %arg117: !torch.vtensor<[64],f32>, %arg118: !torch.vtensor<[64],f32>, %arg119: !torch.vtensor<[64],f32>, %arg120: !torch.vtensor<[64],f32>, %arg121: !torch.vtensor<[384,64,1,1],f32>, %arg122: !torch.vtensor<[384],f32>, %arg123: !torch.vtensor<[384],f32>, %arg124: !torch.vtensor<[384],f32>, %arg125: !torch.vtensor<[384],f32>, %arg126: !torch.vtensor<[384,1,3,3],f32>, %arg127: !torch.vtensor<[384],f32>, %arg128: !torch.vtensor<[384],f32>, %arg129: !torch.vtensor<[384],f32>, %arg130: !torch.vtensor<[384],f32>, %arg131: !torch.vtensor<[64,384,1,1],f32>, %arg132: !torch.vtensor<[64],f32>, %arg133: !torch.vtensor<[64],f32>, %arg134: !torch.vtensor<[64],f32>, %arg135: !torch.vtensor<[64],f32>, %arg136: !torch.vtensor<[384,64,1,1],f32>, %arg137: !torch.vtensor<[384],f32>, %arg138: !torch.vtensor<[384],f32>, %arg139: !torch.vtensor<[384],f32>, %arg140: !torch.vtensor<[384],f32>, %arg141: !torch.vtensor<[384,1,3,3],f32>, %arg142: !torch.vtensor<[384],f32>, %arg143: !torch.vtensor<[384],f32>, %arg144: !torch.vtensor<[384],f32>, %arg145: !torch.vtensor<[384],f32>, %arg146: !torch.vtensor<[64,384,1,1],f32>, %arg147: !torch.vtensor<[64],f32>, %arg148: !torch.vtensor<[64],f32>, %arg149: !torch.vtensor<[64],f32>, %arg150: !torch.vtensor<[64],f32>, %arg151: !torch.vtensor<[384,64,1,1],f32>, %arg152: !torch.vtensor<[384],f32>, %arg153: !torch.vtensor<[384],f32>, %arg154: !torch.vtensor<[384],f32>, %arg155: !torch.vtensor<[384],f32>, %arg156: !torch.vtensor<[384,1,3,3],f32>, %arg157: !torch.vtensor<[384],f32>, %arg158: !torch.vtensor<[384],f32>, %arg159: !torch.vtensor<[384],f32>, %arg160: !torch.vtensor<[384],f32>, %arg161: !torch.vtensor<[96,384,1,1],f32>, %arg162: !torch.vtensor<[96],f32>, %arg163: !torch.vtensor<[96],f32>, %arg164: !torch.vtensor<[96],f32>, %arg165: !torch.vtensor<[96],f32>, %arg166: !torch.vtensor<[576,96,1,1],f32>, %arg167: !torch.vtensor<[576],f32>, %arg168: !torch.vtensor<[576],f32>, %arg169: !torch.vtensor<[576],f32>, %arg170: !torch.vtensor<[576],f32>, %arg171: !torch.vtensor<[576,1,3,3],f32>, %arg172: !torch.vtensor<[576],f32>, %arg173: !torch.vtensor<[576],f32>, %arg174: !torch.vtensor<[576],f32>, %arg175: !torch.vtensor<[576],f32>, %arg176: !torch.vtensor<[96,576,1,1],f32>, %arg177: !torch.vtensor<[96],f32>, %arg178: !torch.vtensor<[96],f32>, %arg179: !torch.vtensor<[96],f32>, %arg180: !torch.vtensor<[96],f32>, %arg181: !torch.vtensor<[576,96,1,1],f32>, %arg182: !torch.vtensor<[576],f32>, %arg183: !torch.vtensor<[576],f32>, %arg184: !torch.vtensor<[576],f32>, %arg185: !torch.vtensor<[576],f32>, %arg186: !torch.vtensor<[576,1,3,3],f32>, %arg187: !torch.vtensor<[576],f32>, %arg188: !torch.vtensor<[576],f32>, %arg189: !torch.vtensor<[576],f32>, %arg190: !torch.vtensor<[576],f32>, %arg191: !torch.vtensor<[96,576,1,1],f32>, %arg192: !torch.vtensor<[96],f32>, %arg193: !torch.vtensor<[96],f32>, %arg194: !torch.vtensor<[96],f32>, %arg195: !torch.vtensor<[96],f32>, %arg196: !torch.vtensor<[576,96,1,1],f32>, %arg197: !torch.vtensor<[576],f32>, %arg198: !torch.vtensor<[576],f32>, %arg199: !torch.vtensor<[576],f32>, %arg200: !torch.vtensor<[576],f32>, %arg201: !torch.vtensor<[576,1,3,3],f32>, %arg202: !torch.vtensor<[576],f32>, %arg203: !torch.vtensor<[576],f32>, %arg204: !torch.vtensor<[576],f32>, %arg205: !torch.vtensor<[576],f32>, %arg206: !torch.vtensor<[160,576,1,1],f32>, %arg207: !torch.vtensor<[160],f32>, %arg208: !torch.vtensor<[160],f32>, %arg209: !torch.vtensor<[160],f32>, %arg210: !torch.vtensor<[160],f32>, %arg211: !torch.vtensor<[960,160,1,1],f32>, %arg212: !torch.vtensor<[960],f32>, %arg213: !torch.vtensor<[960],f32>, %arg214: !torch.vtensor<[960],f32>, %arg215: !torch.vtensor<[960],f32>, %arg216: !torch.vtensor<[960,1,3,3],f32>, %arg217: !torch.vtensor<[960],f32>, %arg218: !torch.vtensor<[960],f32>, %arg219: !torch.vtensor<[960],f32>, %arg220: !torch.vtensor<[960],f32>, %arg221: !torch.vtensor<[160,960,1,1],f32>, %arg222: !torch.vtensor<[160],f32>, %arg223: !torch.vtensor<[160],f32>, %arg224: !torch.vtensor<[160],f32>, %arg225: !torch.vtensor<[160],f32>, %arg226: !torch.vtensor<[960,160,1,1],f32>, %arg227: !torch.vtensor<[960],f32>, %arg228: !torch.vtensor<[960],f32>, %arg229: !torch.vtensor<[960],f32>, %arg230: !torch.vtensor<[960],f32>, %arg231: !torch.vtensor<[960,1,3,3],f32>, %arg232: !torch.vtensor<[960],f32>, %arg233: !torch.vtensor<[960],f32>, %arg234: !torch.vtensor<[960],f32>, %arg235: !torch.vtensor<[960],f32>, %arg236: !torch.vtensor<[160,960,1,1],f32>, %arg237: !torch.vtensor<[160],f32>, %arg238: !torch.vtensor<[160],f32>, %arg239: !torch.vtensor<[160],f32>, %arg240: !torch.vtensor<[160],f32>, %arg241: !torch.vtensor<[960,160,1,1],f32>, %arg242: !torch.vtensor<[960],f32>, %arg243: !torch.vtensor<[960],f32>, %arg244: !torch.vtensor<[960],f32>, %arg245: !torch.vtensor<[960],f32>, %arg246: !torch.vtensor<[960,1,3,3],f32>, %arg247: !torch.vtensor<[960],f32>, %arg248: !torch.vtensor<[960],f32>, %arg249: !torch.vtensor<[960],f32>, %arg250: !torch.vtensor<[960],f32>, %arg251: !torch.vtensor<[320,960,1,1],f32>, %arg252: !torch.vtensor<[320],f32>, %arg253: !torch.vtensor<[320],f32>, %arg254: !torch.vtensor<[320],f32>, %arg255: !torch.vtensor<[320],f32>, %arg256: !torch.vtensor<[1280,320,1,1],f32>, %arg257: !torch.vtensor<[1280],f32>, %arg258: !torch.vtensor<[1280],f32>, %arg259: !torch.vtensor<[1280],f32>, %arg260: !torch.vtensor<[1280],f32>, %arg261: !torch.vtensor<[1001,1280],f32>, %arg262: !torch.vtensor<[1001],f32>) -> (!torch.vtensor<[1,1001],f32>, !torch.vtensor<[32,3,3,3],f32>, !torch.vtensor<[32],f32>, !torch.vtensor<[32],f32>, !torch.vtensor<[32],f32>, !torch.vtensor<[32,1,3,3],f32>, !torch.vtensor<[32],f32>, !torch.vtensor<[32],f32>, !torch.vtensor<[32],f32>, !torch.vtensor<[16,32,1,1],f32>, !torch.vtensor<[16],f32>, !torch.vtensor<[16],f32>, !torch.vtensor<[16],f32>, !torch.vtensor<[96,16,1,1],f32>, !torch.vtensor<[96],f32>, !torch.vtensor<[96],f32>, !torch.vtensor<[96],f32>, !torch.vtensor<[96,1,3,3],f32>, !torch.vtensor<[96],f32>, !torch.vtensor<[96],f32>, !torch.vtensor<[96],f32>, !torch.vtensor<[24,96,1,1],f32>, !torch.vtensor<[24],f32>, !torch.vtensor<[24],f32>, !torch.vtensor<[24],f32>, !torch.vtensor<[144,24,1,1],f32>, !torch.vtensor<[144],f32>, !torch.vtensor<[144],f32>, !torch.vtensor<[144],f32>, !torch.vtensor<[144,1,3,3],f32>, !torch.vtensor<[144],f32>, !torch.vtensor<[144],f32>, !torch.vtensor<[144],f32>, !torch.vtensor<[24,144,1,1],f32>, !torch.vtensor<[24],f32>, !torch.vtensor<[24],f32>, !torch.vtensor<[24],f32>, !torch.vtensor<[144,24,1,1],f32>, !torch.vtensor<[144],f32>, !torch.vtensor<[144],f32>, !torch.vtensor<[144],f32>, !torch.vtensor<[144,1,3,3],f32>, !torch.vtensor<[144],f32>, !torch.vtensor<[144],f32>, !torch.vtensor<[144],f32>, !torch.vtensor<[32,144,1,1],f32>, !torch.vtensor<[32],f32>, !torch.vtensor<[32],f32>, !torch.vtensor<[32],f32>, !torch.vtensor<[192,32,1,1],f32>, !torch.vtensor<[192],f32>, !torch.vtensor<[192],f32>, !torch.vtensor<[192],f32>, !torch.vtensor<[192,1,3,3],f32>, !torch.vtensor<[192],f32>, !torch.vtensor<[192],f32>, !torch.vtensor<[192],f32>, !torch.vtensor<[32,192,1,1],f32>, !torch.vtensor<[32],f32>, !torch.vtensor<[32],f32>, !torch.vtensor<[32],f32>, !torch.vtensor<[192,32,1,1],f32>, !torch.vtensor<[192],f32>, !torch.vtensor<[192],f32>, !torch.vtensor<[192],f32>, !torch.vtensor<[192,1,3,3],f32>, !torch.vtensor<[192],f32>, !torch.vtensor<[192],f32>, !torch.vtensor<[192],f32>, !torch.vtensor<[32,192,1,1],f32>, !torch.vtensor<[32],f32>, !torch.vtensor<[32],f32>, !torch.vtensor<[32],f32>, !torch.vtensor<[192,32,1,1],f32>, !torch.vtensor<[192],f32>, !torch.vtensor<[192],f32>, !torch.vtensor<[192],f32>, !torch.vtensor<[192,1,3,3],f32>, !torch.vtensor<[192],f32>, !torch.vtensor<[192],f32>, !torch.vtensor<[192],f32>, !torch.vtensor<[64,192,1,1],f32>, !torch.vtensor<[64],f32>, !torch.vtensor<[64],f32>, !torch.vtensor<[64],f32>, !torch.vtensor<[384,64,1,1],f32>, !torch.vtensor<[384],f32>, !torch.vtensor<[384],f32>, !torch.vtensor<[384],f32>, !torch.vtensor<[384,1,3,3],f32>, !torch.vtensor<[384],f32>, !torch.vtensor<[384],f32>, !torch.vtensor<[384],f32>, !torch.vtensor<[64,384,1,1],f32>, !torch.vtensor<[64],f32>, !torch.vtensor<[64],f32>, !torch.vtensor<[64],f32>, !torch.vtensor<[384,64,1,1],f32>, !torch.vtensor<[384],f32>, !torch.vtensor<[384],f32>, !torch.vtensor<[384],f32>, !torch.vtensor<[384,1,3,3],f32>, !torch.vtensor<[384],f32>, !torch.vtensor<[384],f32>, !torch.vtensor<[384],f32>, !torch.vtensor<[64,384,1,1],f32>, !torch.vtensor<[64],f32>, !torch.vtensor<[64],f32>, !torch.vtensor<[64],f32>, !torch.vtensor<[384,64,1,1],f32>, !torch.vtensor<[384],f32>, !torch.vtensor<[384],f32>, !torch.vtensor<[384],f32>, !torch.vtensor<[384,1,3,3],f32>, !torch.vtensor<[384],f32>, !torch.vtensor<[384],f32>, !torch.vtensor<[384],f32>, !torch.vtensor<[64,384,1,1],f32>, !torch.vtensor<[64],f32>, !torch.vtensor<[64],f32>, !torch.vtensor<[64],f32>, !torch.vtensor<[384,64,1,1],f32>, !torch.vtensor<[384],f32>, !torch.vtensor<[384],f32>, !torch.vtensor<[384],f32>, !torch.vtensor<[384,1,3,3],f32>, !torch.vtensor<[384],f32>, !torch.vtensor<[384],f32>, !torch.vtensor<[384],f32>, !torch.vtensor<[96,384,1,1],f32>, !torch.vtensor<[96],f32>, !torch.vtensor<[96],f32>, !torch.vtensor<[96],f32>, !torch.vtensor<[576,96,1,1],f32>, !torch.vtensor<[576],f32>, !torch.vtensor<[576],f32>, !torch.vtensor<[576],f32>, !torch.vtensor<[576,1,3,3],f32>, !torch.vtensor<[576],f32>, !torch.vtensor<[576],f32>, !torch.vtensor<[576],f32>, !torch.vtensor<[96,576,1,1],f32>, !torch.vtensor<[96],f32>, !torch.vtensor<[96],f32>, !torch.vtensor<[96],f32>, !torch.vtensor<[576,96,1,1],f32>, !torch.vtensor<[576],f32>, !torch.vtensor<[576],f32>, !torch.vtensor<[576],f32>, !torch.vtensor<[576,1,3,3],f32>, !torch.vtensor<[576],f32>, !torch.vtensor<[576],f32>, !torch.vtensor<[576],f32>, !torch.vtensor<[96,576,1,1],f32>, !torch.vtensor<[96],f32>, !torch.vtensor<[96],f32>, !torch.vtensor<[96],f32>, !torch.vtensor<[576,96,1,1],f32>, !torch.vtensor<[576],f32>, !torch.vtensor<[576],f32>, !torch.vtensor<[576],f32>, !torch.vtensor<[576,1,3,3],f32>, !torch.vtensor<[576],f32>, !torch.vtensor<[576],f32>, !torch.vtensor<[576],f32>, !torch.vtensor<[160,576,1,1],f32>, !torch.vtensor<[160],f32>, !torch.vtensor<[160],f32>, !torch.vtensor<[160],f32>, !torch.vtensor<[960,160,1,1],f32>, !torch.vtensor<[960],f32>, !torch.vtensor<[960],f32>, !torch.vtensor<[960],f32>, !torch.vtensor<[960,1,3,3],f32>, !torch.vtensor<[960],f32>, !torch.vtensor<[960],f32>, !torch.vtensor<[960],f32>, !torch.vtensor<[160,960,1,1],f32>, !torch.vtensor<[160],f32>, !torch.vtensor<[160],f32>, !torch.vtensor<[160],f32>, !torch.vtensor<[960,160,1,1],f32>, !torch.vtensor<[960],f32>, !torch.vtensor<[960],f32>, !torch.vtensor<[960],f32>, !torch.vtensor<[960,1,3,3],f32>, !torch.vtensor<[960],f32>, !torch.vtensor<[960],f32>, !torch.vtensor<[960],f32>, !torch.vtensor<[160,960,1,1],f32>, !torch.vtensor<[160],f32>, !torch.vtensor<[160],f32>, !torch.vtensor<[160],f32>, !torch.vtensor<[960,160,1,1],f32>, !torch.vtensor<[960],f32>, !torch.vtensor<[960],f32>, !torch.vtensor<[960],f32>, !torch.vtensor<[960,1,3,3],f32>, !torch.vtensor<[960],f32>, !torch.vtensor<[960],f32>, !torch.vtensor<[960],f32>, !torch.vtensor<[320,960,1,1],f32>, !torch.vtensor<[320],f32>, !torch.vtensor<[320],f32>, !torch.vtensor<[320],f32>, !torch.vtensor<[1280,320,1,1],f32>, !torch.vtensor<[1280],f32>, !torch.vtensor<[1280],f32>, !torch.vtensor<[1280],f32>, !torch.vtensor<[1,3,225,225],f32>, !torch.vtensor<[1,32,112,112],f32>, !torch.vtensor<[1,32,112,112],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[1,32,114,114],f32>, !torch.vtensor<[1,32,112,112],f32>, !torch.vtensor<[1,32,112,112],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[1,32,112,112],f32>, !torch.vtensor<[1,16,112,112],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[1,16,112,112],f32>, !torch.vtensor<[1,96,112,112],f32>, !torch.vtensor<[1,96,112,112],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[1,96,113,113],f32>, !torch.vtensor<[1,96,56,56],f32>, !torch.vtensor<[1,96,56,56],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[1,96,56,56],f32>, !torch.vtensor<[1,24,56,56],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[1,24,56,56],f32>, !torch.vtensor<[1,144,56,56],f32>, !torch.vtensor<[1,144,56,56],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[1,144,58,58],f32>, !torch.vtensor<[1,144,56,56],f32>, !torch.vtensor<[1,144,56,56],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[1,144,56,56],f32>, !torch.vtensor<[1,24,56,56],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[1,24,56,56],f32>, !torch.vtensor<[1,144,56,56],f32>, !torch.vtensor<[1,144,56,56],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[1,144,57,57],f32>, !torch.vtensor<[1,144,28,28],f32>, !torch.vtensor<[1,144,28,28],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[1,144,28,28],f32>, !torch.vtensor<[1,32,28,28],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[1,32,28,28],f32>, !torch.vtensor<[1,192,28,28],f32>, !torch.vtensor<[1,192,28,28],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[1,192,30,30],f32>, !torch.vtensor<[1,192,28,28],f32>, !torch.vtensor<[1,192,28,28],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[1,192,28,28],f32>, !torch.vtensor<[1,32,28,28],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[1,32,28,28],f32>, !torch.vtensor<[1,192,28,28],f32>, !torch.vtensor<[1,192,28,28],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[1,192,30,30],f32>, !torch.vtensor<[1,192,28,28],f32>, !torch.vtensor<[1,192,28,28],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[1,192,28,28],f32>, !torch.vtensor<[1,32,28,28],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[1,32,28,28],f32>, !torch.vtensor<[1,192,28,28],f32>, !torch.vtensor<[1,192,28,28],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[1,192,29,29],f32>, !torch.vtensor<[1,192,14,14],f32>, !torch.vtensor<[1,192,14,14],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[1,192,14,14],f32>, !torch.vtensor<[1,64,14,14],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[1,64,14,14],f32>, !torch.vtensor<[1,384,14,14],f32>, !torch.vtensor<[1,384,14,14],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[1,384,16,16],f32>, !torch.vtensor<[1,384,14,14],f32>, !torch.vtensor<[1,384,14,14],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[1,384,14,14],f32>, !torch.vtensor<[1,64,14,14],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[1,64,14,14],f32>, !torch.vtensor<[1,384,14,14],f32>, !torch.vtensor<[1,384,14,14],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[1,384,16,16],f32>, !torch.vtensor<[1,384,14,14],f32>, !torch.vtensor<[1,384,14,14],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[1,384,14,14],f32>, !torch.vtensor<[1,64,14,14],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[1,64,14,14],f32>, !torch.vtensor<[1,384,14,14],f32>, !torch.vtensor<[1,384,14,14],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[1,384,16,16],f32>, !torch.vtensor<[1,384,14,14],f32>, !torch.vtensor<[1,384,14,14],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[1,384,14,14],f32>, !torch.vtensor<[1,64,14,14],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[1,64,14,14],f32>, !torch.vtensor<[1,384,14,14],f32>, !torch.vtensor<[1,384,14,14],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[1,384,16,16],f32>, !torch.vtensor<[1,384,14,14],f32>, !torch.vtensor<[1,384,14,14],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[1,384,14,14],f32>, !torch.vtensor<[1,96,14,14],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[1,96,14,14],f32>, !torch.vtensor<[1,576,14,14],f32>, !torch.vtensor<[1,576,14,14],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[1,576,16,16],f32>, !torch.vtensor<[1,576,14,14],f32>, !torch.vtensor<[1,576,14,14],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[1,576,14,14],f32>, !torch.vtensor<[1,96,14,14],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[1,96,14,14],f32>, !torch.vtensor<[1,576,14,14],f32>, !torch.vtensor<[1,576,14,14],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[1,576,16,16],f32>, !torch.vtensor<[1,576,14,14],f32>, !torch.vtensor<[1,576,14,14],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[1,576,14,14],f32>, !torch.vtensor<[1,96,14,14],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[1,96,14,14],f32>, !torch.vtensor<[1,576,14,14],f32>, !torch.vtensor<[1,576,14,14],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[1,576,15,15],f32>, !torch.vtensor<[1,576,7,7],f32>, !torch.vtensor<[1,576,7,7],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[1,576,7,7],f32>, !torch.vtensor<[1,160,7,7],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[1,160,7,7],f32>, !torch.vtensor<[1,960,7,7],f32>, !torch.vtensor<[1,960,7,7],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[1,960,9,9],f32>, !torch.vtensor<[1,960,7,7],f32>, !torch.vtensor<[1,960,7,7],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[1,960,7,7],f32>, !torch.vtensor<[1,160,7,7],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[1,160,7,7],f32>, !torch.vtensor<[1,960,7,7],f32>, !torch.vtensor<[1,960,7,7],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[1,960,9,9],f32>, !torch.vtensor<[1,960,7,7],f32>, !torch.vtensor<[1,960,7,7],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[1,960,7,7],f32>, !torch.vtensor<[1,160,7,7],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[1,160,7,7],f32>, !torch.vtensor<[1,960,7,7],f32>, !torch.vtensor<[1,960,7,7],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[1,960,9,9],f32>, !torch.vtensor<[1,960,7,7],f32>, !torch.vtensor<[1,960,7,7],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[1,960,7,7],f32>, !torch.vtensor<[1,320,7,7],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[1,320,7,7],f32>, !torch.vtensor<[1,1280,7,7],f32>, !torch.vtensor<[1,1280,7,7],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[1,1280],f32>, !torch.vtensor<[1280,1001],f32>) {
    %int0 = torch.constant.int 0
    %int1 = torch.constant.int 1
    %int0_0 = torch.constant.int 0
    %int1_1 = torch.constant.int 1
    %0 = torch.prim.ListConstruct %int0, %int1, %int0_0, %int1_1 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %float0.000000e00 = torch.constant.float 0.000000e+00
    %1 = torch.aten.constant_pad_nd %arg0, %0, %float0.000000e00 : !torch.vtensor<[1,3,224,224],f32>, !torch.list<int>, !torch.float -> !torch.vtensor<[1,3,225,225],f32>
    %none = torch.constant.none
    %int2 = torch.constant.int 2
    %int2_2 = torch.constant.int 2
    %2 = torch.prim.ListConstruct %int2, %int2_2 : (!torch.int, !torch.int) -> !torch.list<int>
    %int0_3 = torch.constant.int 0
    %int0_4 = torch.constant.int 0
    %3 = torch.prim.ListConstruct %int0_3, %int0_4 : (!torch.int, !torch.int) -> !torch.list<int>
    %int1_5 = torch.constant.int 1
    %int1_6 = torch.constant.int 1
    %4 = torch.prim.ListConstruct %int1_5, %int1_6 : (!torch.int, !torch.int) -> !torch.list<int>
    %false = torch.constant.bool false
    %int0_7 = torch.constant.int 0
    %int0_8 = torch.constant.int 0
    %5 = torch.prim.ListConstruct %int0_7, %int0_8 : (!torch.int, !torch.int) -> !torch.list<int>
    %int1_9 = torch.constant.int 1
    %6 = torch.aten.convolution %1, %arg1, %none, %2, %3, %4, %false, %5, %int1_9 : !torch.vtensor<[1,3,225,225],f32>, !torch.vtensor<[32,3,3,3],f32>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[1,32,112,112],f32>
    %int6 = torch.constant.int 6
    %7 = torch.prims.convert_element_type %arg2, %int6 : !torch.vtensor<[32],f32>, !torch.int -> !torch.vtensor<[32],f32>
    %int6_10 = torch.constant.int 6
    %8 = torch.prims.convert_element_type %arg3, %int6_10 : !torch.vtensor<[32],f32>, !torch.int -> !torch.vtensor<[32],f32>
    %float1.000000e-03 = torch.constant.float 1.000000e-03
    %int1_11 = torch.constant.int 1
    %9 = torch.aten.add.Scalar %8, %float1.000000e-03, %int1_11 : !torch.vtensor<[32],f32>, !torch.float, !torch.int -> !torch.vtensor<[32],f32>
    %10 = torch.aten.sqrt %9 : !torch.vtensor<[32],f32> -> !torch.vtensor<[32],f32>
    %11 = torch.aten.reciprocal %10 : !torch.vtensor<[32],f32> -> !torch.vtensor<[32],f32>
    %int1_12 = torch.constant.int 1
    %12 = torch.aten.mul.Scalar %11, %int1_12 : !torch.vtensor<[32],f32>, !torch.int -> !torch.vtensor<[32],f32>
    %int0_13 = torch.constant.int 0
    %13 = torch.prim.ListConstruct %int0_13 : (!torch.int) -> !torch.list<int>
    %none_14 = torch.constant.none
    %none_15 = torch.constant.none
    %none_16 = torch.constant.none
    %false_17 = torch.constant.bool false
    %14 = torch.aten.new_zeros %6, %13, %none_14, %none_15, %none_16, %false_17 : !torch.vtensor<[1,32,112,112],f32>, !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[0],f32>
    %int0_18 = torch.constant.int 0
    %15 = torch.prim.ListConstruct %int0_18 : (!torch.int) -> !torch.list<int>
    %none_19 = torch.constant.none
    %none_20 = torch.constant.none
    %none_21 = torch.constant.none
    %false_22 = torch.constant.bool false
    %16 = torch.aten.new_zeros %6, %15, %none_19, %none_20, %none_21, %false_22 : !torch.vtensor<[1,32,112,112],f32>, !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[0],f32>
    %int-1 = torch.constant.int -1
    %17 = torch.aten.unsqueeze %7, %int-1 : !torch.vtensor<[32],f32>, !torch.int -> !torch.vtensor<[32,1],f32>
    %int-1_23 = torch.constant.int -1
    %18 = torch.aten.unsqueeze %17, %int-1_23 : !torch.vtensor<[32,1],f32>, !torch.int -> !torch.vtensor<[32,1,1],f32>
    %int-1_24 = torch.constant.int -1
    %19 = torch.aten.unsqueeze %12, %int-1_24 : !torch.vtensor<[32],f32>, !torch.int -> !torch.vtensor<[32,1],f32>
    %int-1_25 = torch.constant.int -1
    %20 = torch.aten.unsqueeze %19, %int-1_25 : !torch.vtensor<[32,1],f32>, !torch.int -> !torch.vtensor<[32,1,1],f32>
    %int1_26 = torch.constant.int 1
    %21 = torch.aten.sub.Tensor %6, %18, %int1_26 : !torch.vtensor<[1,32,112,112],f32>, !torch.vtensor<[32,1,1],f32>, !torch.int -> !torch.vtensor<[1,32,112,112],f32>
    %22 = torch.aten.mul.Tensor %21, %20 : !torch.vtensor<[1,32,112,112],f32>, !torch.vtensor<[32,1,1],f32> -> !torch.vtensor<[1,32,112,112],f32>
    %int-1_27 = torch.constant.int -1
    %23 = torch.aten.unsqueeze %arg4, %int-1_27 : !torch.vtensor<[32],f32>, !torch.int -> !torch.vtensor<[32,1],f32>
    %int-1_28 = torch.constant.int -1
    %24 = torch.aten.unsqueeze %23, %int-1_28 : !torch.vtensor<[32,1],f32>, !torch.int -> !torch.vtensor<[32,1,1],f32>
    %25 = torch.aten.mul.Tensor %22, %24 : !torch.vtensor<[1,32,112,112],f32>, !torch.vtensor<[32,1,1],f32> -> !torch.vtensor<[1,32,112,112],f32>
    %int-1_29 = torch.constant.int -1
    %26 = torch.aten.unsqueeze %arg5, %int-1_29 : !torch.vtensor<[32],f32>, !torch.int -> !torch.vtensor<[32,1],f32>
    %int-1_30 = torch.constant.int -1
    %27 = torch.aten.unsqueeze %26, %int-1_30 : !torch.vtensor<[32,1],f32>, !torch.int -> !torch.vtensor<[32,1,1],f32>
    %int1_31 = torch.constant.int 1
    %28 = torch.aten.add.Tensor %25, %27, %int1_31 : !torch.vtensor<[1,32,112,112],f32>, !torch.vtensor<[32,1,1],f32>, !torch.int -> !torch.vtensor<[1,32,112,112],f32>
    %float0.000000e00_32 = torch.constant.float 0.000000e+00
    %float6.000000e00 = torch.constant.float 6.000000e+00
    %29 = torch.aten.hardtanh %28, %float0.000000e00_32, %float6.000000e00 : !torch.vtensor<[1,32,112,112],f32>, !torch.float, !torch.float -> !torch.vtensor<[1,32,112,112],f32>
    %int1_33 = torch.constant.int 1
    %int1_34 = torch.constant.int 1
    %int1_35 = torch.constant.int 1
    %int1_36 = torch.constant.int 1
    %30 = torch.prim.ListConstruct %int1_33, %int1_34, %int1_35, %int1_36 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %float0.000000e00_37 = torch.constant.float 0.000000e+00
    %31 = torch.aten.constant_pad_nd %29, %30, %float0.000000e00_37 : !torch.vtensor<[1,32,112,112],f32>, !torch.list<int>, !torch.float -> !torch.vtensor<[1,32,114,114],f32>
    %none_38 = torch.constant.none
    %int1_39 = torch.constant.int 1
    %int1_40 = torch.constant.int 1
    %32 = torch.prim.ListConstruct %int1_39, %int1_40 : (!torch.int, !torch.int) -> !torch.list<int>
    %int0_41 = torch.constant.int 0
    %int0_42 = torch.constant.int 0
    %33 = torch.prim.ListConstruct %int0_41, %int0_42 : (!torch.int, !torch.int) -> !torch.list<int>
    %int1_43 = torch.constant.int 1
    %int1_44 = torch.constant.int 1
    %34 = torch.prim.ListConstruct %int1_43, %int1_44 : (!torch.int, !torch.int) -> !torch.list<int>
    %false_45 = torch.constant.bool false
    %int0_46 = torch.constant.int 0
    %int0_47 = torch.constant.int 0
    %35 = torch.prim.ListConstruct %int0_46, %int0_47 : (!torch.int, !torch.int) -> !torch.list<int>
    %int32 = torch.constant.int 32
    %36 = torch.aten.convolution %31, %arg6, %none_38, %32, %33, %34, %false_45, %35, %int32 : !torch.vtensor<[1,32,114,114],f32>, !torch.vtensor<[32,1,3,3],f32>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[1,32,112,112],f32>
    %int6_48 = torch.constant.int 6
    %37 = torch.prims.convert_element_type %arg7, %int6_48 : !torch.vtensor<[32],f32>, !torch.int -> !torch.vtensor<[32],f32>
    %int6_49 = torch.constant.int 6
    %38 = torch.prims.convert_element_type %arg8, %int6_49 : !torch.vtensor<[32],f32>, !torch.int -> !torch.vtensor<[32],f32>
    %float1.000000e-03_50 = torch.constant.float 1.000000e-03
    %int1_51 = torch.constant.int 1
    %39 = torch.aten.add.Scalar %38, %float1.000000e-03_50, %int1_51 : !torch.vtensor<[32],f32>, !torch.float, !torch.int -> !torch.vtensor<[32],f32>
    %40 = torch.aten.sqrt %39 : !torch.vtensor<[32],f32> -> !torch.vtensor<[32],f32>
    %41 = torch.aten.reciprocal %40 : !torch.vtensor<[32],f32> -> !torch.vtensor<[32],f32>
    %int1_52 = torch.constant.int 1
    %42 = torch.aten.mul.Scalar %41, %int1_52 : !torch.vtensor<[32],f32>, !torch.int -> !torch.vtensor<[32],f32>
    %int0_53 = torch.constant.int 0
    %43 = torch.prim.ListConstruct %int0_53 : (!torch.int) -> !torch.list<int>
    %none_54 = torch.constant.none
    %none_55 = torch.constant.none
    %none_56 = torch.constant.none
    %false_57 = torch.constant.bool false
    %44 = torch.aten.new_zeros %36, %43, %none_54, %none_55, %none_56, %false_57 : !torch.vtensor<[1,32,112,112],f32>, !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[0],f32>
    %int0_58 = torch.constant.int 0
    %45 = torch.prim.ListConstruct %int0_58 : (!torch.int) -> !torch.list<int>
    %none_59 = torch.constant.none
    %none_60 = torch.constant.none
    %none_61 = torch.constant.none
    %false_62 = torch.constant.bool false
    %46 = torch.aten.new_zeros %36, %45, %none_59, %none_60, %none_61, %false_62 : !torch.vtensor<[1,32,112,112],f32>, !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[0],f32>
    %int-1_63 = torch.constant.int -1
    %47 = torch.aten.unsqueeze %37, %int-1_63 : !torch.vtensor<[32],f32>, !torch.int -> !torch.vtensor<[32,1],f32>
    %int-1_64 = torch.constant.int -1
    %48 = torch.aten.unsqueeze %47, %int-1_64 : !torch.vtensor<[32,1],f32>, !torch.int -> !torch.vtensor<[32,1,1],f32>
    %int-1_65 = torch.constant.int -1
    %49 = torch.aten.unsqueeze %42, %int-1_65 : !torch.vtensor<[32],f32>, !torch.int -> !torch.vtensor<[32,1],f32>
    %int-1_66 = torch.constant.int -1
    %50 = torch.aten.unsqueeze %49, %int-1_66 : !torch.vtensor<[32,1],f32>, !torch.int -> !torch.vtensor<[32,1,1],f32>
    %int1_67 = torch.constant.int 1
    %51 = torch.aten.sub.Tensor %36, %48, %int1_67 : !torch.vtensor<[1,32,112,112],f32>, !torch.vtensor<[32,1,1],f32>, !torch.int -> !torch.vtensor<[1,32,112,112],f32>
    %52 = torch.aten.mul.Tensor %51, %50 : !torch.vtensor<[1,32,112,112],f32>, !torch.vtensor<[32,1,1],f32> -> !torch.vtensor<[1,32,112,112],f32>
    %int-1_68 = torch.constant.int -1
    %53 = torch.aten.unsqueeze %arg9, %int-1_68 : !torch.vtensor<[32],f32>, !torch.int -> !torch.vtensor<[32,1],f32>
    %int-1_69 = torch.constant.int -1
    %54 = torch.aten.unsqueeze %53, %int-1_69 : !torch.vtensor<[32,1],f32>, !torch.int -> !torch.vtensor<[32,1,1],f32>
    %55 = torch.aten.mul.Tensor %52, %54 : !torch.vtensor<[1,32,112,112],f32>, !torch.vtensor<[32,1,1],f32> -> !torch.vtensor<[1,32,112,112],f32>
    %int-1_70 = torch.constant.int -1
    %56 = torch.aten.unsqueeze %arg10, %int-1_70 : !torch.vtensor<[32],f32>, !torch.int -> !torch.vtensor<[32,1],f32>
    %int-1_71 = torch.constant.int -1
    %57 = torch.aten.unsqueeze %56, %int-1_71 : !torch.vtensor<[32,1],f32>, !torch.int -> !torch.vtensor<[32,1,1],f32>
    %int1_72 = torch.constant.int 1
    %58 = torch.aten.add.Tensor %55, %57, %int1_72 : !torch.vtensor<[1,32,112,112],f32>, !torch.vtensor<[32,1,1],f32>, !torch.int -> !torch.vtensor<[1,32,112,112],f32>
    %float0.000000e00_73 = torch.constant.float 0.000000e+00
    %float6.000000e00_74 = torch.constant.float 6.000000e+00
    %59 = torch.aten.hardtanh %58, %float0.000000e00_73, %float6.000000e00_74 : !torch.vtensor<[1,32,112,112],f32>, !torch.float, !torch.float -> !torch.vtensor<[1,32,112,112],f32>
    %int0_75 = torch.constant.int 0
    %int0_76 = torch.constant.int 0
    %int0_77 = torch.constant.int 0
    %int0_78 = torch.constant.int 0
    %60 = torch.prim.ListConstruct %int0_75, %int0_76, %int0_77, %int0_78 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %float0.000000e00_79 = torch.constant.float 0.000000e+00
    %61 = torch.aten.constant_pad_nd %59, %60, %float0.000000e00_79 : !torch.vtensor<[1,32,112,112],f32>, !torch.list<int>, !torch.float -> !torch.vtensor<[1,32,112,112],f32>
    %none_80 = torch.constant.none
    %int1_81 = torch.constant.int 1
    %int1_82 = torch.constant.int 1
    %62 = torch.prim.ListConstruct %int1_81, %int1_82 : (!torch.int, !torch.int) -> !torch.list<int>
    %int0_83 = torch.constant.int 0
    %int0_84 = torch.constant.int 0
    %63 = torch.prim.ListConstruct %int0_83, %int0_84 : (!torch.int, !torch.int) -> !torch.list<int>
    %int1_85 = torch.constant.int 1
    %int1_86 = torch.constant.int 1
    %64 = torch.prim.ListConstruct %int1_85, %int1_86 : (!torch.int, !torch.int) -> !torch.list<int>
    %false_87 = torch.constant.bool false
    %int0_88 = torch.constant.int 0
    %int0_89 = torch.constant.int 0
    %65 = torch.prim.ListConstruct %int0_88, %int0_89 : (!torch.int, !torch.int) -> !torch.list<int>
    %int1_90 = torch.constant.int 1
    %66 = torch.aten.convolution %61, %arg11, %none_80, %62, %63, %64, %false_87, %65, %int1_90 : !torch.vtensor<[1,32,112,112],f32>, !torch.vtensor<[16,32,1,1],f32>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[1,16,112,112],f32>
    %int6_91 = torch.constant.int 6
    %67 = torch.prims.convert_element_type %arg12, %int6_91 : !torch.vtensor<[16],f32>, !torch.int -> !torch.vtensor<[16],f32>
    %int6_92 = torch.constant.int 6
    %68 = torch.prims.convert_element_type %arg13, %int6_92 : !torch.vtensor<[16],f32>, !torch.int -> !torch.vtensor<[16],f32>
    %float1.000000e-03_93 = torch.constant.float 1.000000e-03
    %int1_94 = torch.constant.int 1
    %69 = torch.aten.add.Scalar %68, %float1.000000e-03_93, %int1_94 : !torch.vtensor<[16],f32>, !torch.float, !torch.int -> !torch.vtensor<[16],f32>
    %70 = torch.aten.sqrt %69 : !torch.vtensor<[16],f32> -> !torch.vtensor<[16],f32>
    %71 = torch.aten.reciprocal %70 : !torch.vtensor<[16],f32> -> !torch.vtensor<[16],f32>
    %int1_95 = torch.constant.int 1
    %72 = torch.aten.mul.Scalar %71, %int1_95 : !torch.vtensor<[16],f32>, !torch.int -> !torch.vtensor<[16],f32>
    %int0_96 = torch.constant.int 0
    %73 = torch.prim.ListConstruct %int0_96 : (!torch.int) -> !torch.list<int>
    %none_97 = torch.constant.none
    %none_98 = torch.constant.none
    %none_99 = torch.constant.none
    %false_100 = torch.constant.bool false
    %74 = torch.aten.new_zeros %66, %73, %none_97, %none_98, %none_99, %false_100 : !torch.vtensor<[1,16,112,112],f32>, !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[0],f32>
    %int0_101 = torch.constant.int 0
    %75 = torch.prim.ListConstruct %int0_101 : (!torch.int) -> !torch.list<int>
    %none_102 = torch.constant.none
    %none_103 = torch.constant.none
    %none_104 = torch.constant.none
    %false_105 = torch.constant.bool false
    %76 = torch.aten.new_zeros %66, %75, %none_102, %none_103, %none_104, %false_105 : !torch.vtensor<[1,16,112,112],f32>, !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[0],f32>
    %int-1_106 = torch.constant.int -1
    %77 = torch.aten.unsqueeze %67, %int-1_106 : !torch.vtensor<[16],f32>, !torch.int -> !torch.vtensor<[16,1],f32>
    %int-1_107 = torch.constant.int -1
    %78 = torch.aten.unsqueeze %77, %int-1_107 : !torch.vtensor<[16,1],f32>, !torch.int -> !torch.vtensor<[16,1,1],f32>
    %int-1_108 = torch.constant.int -1
    %79 = torch.aten.unsqueeze %72, %int-1_108 : !torch.vtensor<[16],f32>, !torch.int -> !torch.vtensor<[16,1],f32>
    %int-1_109 = torch.constant.int -1
    %80 = torch.aten.unsqueeze %79, %int-1_109 : !torch.vtensor<[16,1],f32>, !torch.int -> !torch.vtensor<[16,1,1],f32>
    %int1_110 = torch.constant.int 1
    %81 = torch.aten.sub.Tensor %66, %78, %int1_110 : !torch.vtensor<[1,16,112,112],f32>, !torch.vtensor<[16,1,1],f32>, !torch.int -> !torch.vtensor<[1,16,112,112],f32>
    %82 = torch.aten.mul.Tensor %81, %80 : !torch.vtensor<[1,16,112,112],f32>, !torch.vtensor<[16,1,1],f32> -> !torch.vtensor<[1,16,112,112],f32>
    %int-1_111 = torch.constant.int -1
    %83 = torch.aten.unsqueeze %arg14, %int-1_111 : !torch.vtensor<[16],f32>, !torch.int -> !torch.vtensor<[16,1],f32>
    %int-1_112 = torch.constant.int -1
    %84 = torch.aten.unsqueeze %83, %int-1_112 : !torch.vtensor<[16,1],f32>, !torch.int -> !torch.vtensor<[16,1,1],f32>
    %85 = torch.aten.mul.Tensor %82, %84 : !torch.vtensor<[1,16,112,112],f32>, !torch.vtensor<[16,1,1],f32> -> !torch.vtensor<[1,16,112,112],f32>
    %int-1_113 = torch.constant.int -1
    %86 = torch.aten.unsqueeze %arg15, %int-1_113 : !torch.vtensor<[16],f32>, !torch.int -> !torch.vtensor<[16,1],f32>
    %int-1_114 = torch.constant.int -1
    %87 = torch.aten.unsqueeze %86, %int-1_114 : !torch.vtensor<[16,1],f32>, !torch.int -> !torch.vtensor<[16,1,1],f32>
    %int1_115 = torch.constant.int 1
    %88 = torch.aten.add.Tensor %85, %87, %int1_115 : !torch.vtensor<[1,16,112,112],f32>, !torch.vtensor<[16,1,1],f32>, !torch.int -> !torch.vtensor<[1,16,112,112],f32>
    %int0_116 = torch.constant.int 0
    %int0_117 = torch.constant.int 0
    %int0_118 = torch.constant.int 0
    %int0_119 = torch.constant.int 0
    %89 = torch.prim.ListConstruct %int0_116, %int0_117, %int0_118, %int0_119 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %float0.000000e00_120 = torch.constant.float 0.000000e+00
    %90 = torch.aten.constant_pad_nd %88, %89, %float0.000000e00_120 : !torch.vtensor<[1,16,112,112],f32>, !torch.list<int>, !torch.float -> !torch.vtensor<[1,16,112,112],f32>
    %none_121 = torch.constant.none
    %int1_122 = torch.constant.int 1
    %int1_123 = torch.constant.int 1
    %91 = torch.prim.ListConstruct %int1_122, %int1_123 : (!torch.int, !torch.int) -> !torch.list<int>
    %int0_124 = torch.constant.int 0
    %int0_125 = torch.constant.int 0
    %92 = torch.prim.ListConstruct %int0_124, %int0_125 : (!torch.int, !torch.int) -> !torch.list<int>
    %int1_126 = torch.constant.int 1
    %int1_127 = torch.constant.int 1
    %93 = torch.prim.ListConstruct %int1_126, %int1_127 : (!torch.int, !torch.int) -> !torch.list<int>
    %false_128 = torch.constant.bool false
    %int0_129 = torch.constant.int 0
    %int0_130 = torch.constant.int 0
    %94 = torch.prim.ListConstruct %int0_129, %int0_130 : (!torch.int, !torch.int) -> !torch.list<int>
    %int1_131 = torch.constant.int 1
    %95 = torch.aten.convolution %90, %arg16, %none_121, %91, %92, %93, %false_128, %94, %int1_131 : !torch.vtensor<[1,16,112,112],f32>, !torch.vtensor<[96,16,1,1],f32>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[1,96,112,112],f32>
    %int6_132 = torch.constant.int 6
    %96 = torch.prims.convert_element_type %arg17, %int6_132 : !torch.vtensor<[96],f32>, !torch.int -> !torch.vtensor<[96],f32>
    %int6_133 = torch.constant.int 6
    %97 = torch.prims.convert_element_type %arg18, %int6_133 : !torch.vtensor<[96],f32>, !torch.int -> !torch.vtensor<[96],f32>
    %float1.000000e-03_134 = torch.constant.float 1.000000e-03
    %int1_135 = torch.constant.int 1
    %98 = torch.aten.add.Scalar %97, %float1.000000e-03_134, %int1_135 : !torch.vtensor<[96],f32>, !torch.float, !torch.int -> !torch.vtensor<[96],f32>
    %99 = torch.aten.sqrt %98 : !torch.vtensor<[96],f32> -> !torch.vtensor<[96],f32>
    %100 = torch.aten.reciprocal %99 : !torch.vtensor<[96],f32> -> !torch.vtensor<[96],f32>
    %int1_136 = torch.constant.int 1
    %101 = torch.aten.mul.Scalar %100, %int1_136 : !torch.vtensor<[96],f32>, !torch.int -> !torch.vtensor<[96],f32>
    %int0_137 = torch.constant.int 0
    %102 = torch.prim.ListConstruct %int0_137 : (!torch.int) -> !torch.list<int>
    %none_138 = torch.constant.none
    %none_139 = torch.constant.none
    %none_140 = torch.constant.none
    %false_141 = torch.constant.bool false
    %103 = torch.aten.new_zeros %95, %102, %none_138, %none_139, %none_140, %false_141 : !torch.vtensor<[1,96,112,112],f32>, !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[0],f32>
    %int0_142 = torch.constant.int 0
    %104 = torch.prim.ListConstruct %int0_142 : (!torch.int) -> !torch.list<int>
    %none_143 = torch.constant.none
    %none_144 = torch.constant.none
    %none_145 = torch.constant.none
    %false_146 = torch.constant.bool false
    %105 = torch.aten.new_zeros %95, %104, %none_143, %none_144, %none_145, %false_146 : !torch.vtensor<[1,96,112,112],f32>, !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[0],f32>
    %int-1_147 = torch.constant.int -1
    %106 = torch.aten.unsqueeze %96, %int-1_147 : !torch.vtensor<[96],f32>, !torch.int -> !torch.vtensor<[96,1],f32>
    %int-1_148 = torch.constant.int -1
    %107 = torch.aten.unsqueeze %106, %int-1_148 : !torch.vtensor<[96,1],f32>, !torch.int -> !torch.vtensor<[96,1,1],f32>
    %int-1_149 = torch.constant.int -1
    %108 = torch.aten.unsqueeze %101, %int-1_149 : !torch.vtensor<[96],f32>, !torch.int -> !torch.vtensor<[96,1],f32>
    %int-1_150 = torch.constant.int -1
    %109 = torch.aten.unsqueeze %108, %int-1_150 : !torch.vtensor<[96,1],f32>, !torch.int -> !torch.vtensor<[96,1,1],f32>
    %int1_151 = torch.constant.int 1
    %110 = torch.aten.sub.Tensor %95, %107, %int1_151 : !torch.vtensor<[1,96,112,112],f32>, !torch.vtensor<[96,1,1],f32>, !torch.int -> !torch.vtensor<[1,96,112,112],f32>
    %111 = torch.aten.mul.Tensor %110, %109 : !torch.vtensor<[1,96,112,112],f32>, !torch.vtensor<[96,1,1],f32> -> !torch.vtensor<[1,96,112,112],f32>
    %int-1_152 = torch.constant.int -1
    %112 = torch.aten.unsqueeze %arg19, %int-1_152 : !torch.vtensor<[96],f32>, !torch.int -> !torch.vtensor<[96,1],f32>
    %int-1_153 = torch.constant.int -1
    %113 = torch.aten.unsqueeze %112, %int-1_153 : !torch.vtensor<[96,1],f32>, !torch.int -> !torch.vtensor<[96,1,1],f32>
    %114 = torch.aten.mul.Tensor %111, %113 : !torch.vtensor<[1,96,112,112],f32>, !torch.vtensor<[96,1,1],f32> -> !torch.vtensor<[1,96,112,112],f32>
    %int-1_154 = torch.constant.int -1
    %115 = torch.aten.unsqueeze %arg20, %int-1_154 : !torch.vtensor<[96],f32>, !torch.int -> !torch.vtensor<[96,1],f32>
    %int-1_155 = torch.constant.int -1
    %116 = torch.aten.unsqueeze %115, %int-1_155 : !torch.vtensor<[96,1],f32>, !torch.int -> !torch.vtensor<[96,1,1],f32>
    %int1_156 = torch.constant.int 1
    %117 = torch.aten.add.Tensor %114, %116, %int1_156 : !torch.vtensor<[1,96,112,112],f32>, !torch.vtensor<[96,1,1],f32>, !torch.int -> !torch.vtensor<[1,96,112,112],f32>
    %float0.000000e00_157 = torch.constant.float 0.000000e+00
    %float6.000000e00_158 = torch.constant.float 6.000000e+00
    %118 = torch.aten.hardtanh %117, %float0.000000e00_157, %float6.000000e00_158 : !torch.vtensor<[1,96,112,112],f32>, !torch.float, !torch.float -> !torch.vtensor<[1,96,112,112],f32>
    %int0_159 = torch.constant.int 0
    %int1_160 = torch.constant.int 1
    %int0_161 = torch.constant.int 0
    %int1_162 = torch.constant.int 1
    %119 = torch.prim.ListConstruct %int0_159, %int1_160, %int0_161, %int1_162 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %float0.000000e00_163 = torch.constant.float 0.000000e+00
    %120 = torch.aten.constant_pad_nd %118, %119, %float0.000000e00_163 : !torch.vtensor<[1,96,112,112],f32>, !torch.list<int>, !torch.float -> !torch.vtensor<[1,96,113,113],f32>
    %none_164 = torch.constant.none
    %int2_165 = torch.constant.int 2
    %int2_166 = torch.constant.int 2
    %121 = torch.prim.ListConstruct %int2_165, %int2_166 : (!torch.int, !torch.int) -> !torch.list<int>
    %int0_167 = torch.constant.int 0
    %int0_168 = torch.constant.int 0
    %122 = torch.prim.ListConstruct %int0_167, %int0_168 : (!torch.int, !torch.int) -> !torch.list<int>
    %int1_169 = torch.constant.int 1
    %int1_170 = torch.constant.int 1
    %123 = torch.prim.ListConstruct %int1_169, %int1_170 : (!torch.int, !torch.int) -> !torch.list<int>
    %false_171 = torch.constant.bool false
    %int0_172 = torch.constant.int 0
    %int0_173 = torch.constant.int 0
    %124 = torch.prim.ListConstruct %int0_172, %int0_173 : (!torch.int, !torch.int) -> !torch.list<int>
    %int96 = torch.constant.int 96
    %125 = torch.aten.convolution %120, %arg21, %none_164, %121, %122, %123, %false_171, %124, %int96 : !torch.vtensor<[1,96,113,113],f32>, !torch.vtensor<[96,1,3,3],f32>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[1,96,56,56],f32>
    %int6_174 = torch.constant.int 6
    %126 = torch.prims.convert_element_type %arg22, %int6_174 : !torch.vtensor<[96],f32>, !torch.int -> !torch.vtensor<[96],f32>
    %int6_175 = torch.constant.int 6
    %127 = torch.prims.convert_element_type %arg23, %int6_175 : !torch.vtensor<[96],f32>, !torch.int -> !torch.vtensor<[96],f32>
    %float1.000000e-03_176 = torch.constant.float 1.000000e-03
    %int1_177 = torch.constant.int 1
    %128 = torch.aten.add.Scalar %127, %float1.000000e-03_176, %int1_177 : !torch.vtensor<[96],f32>, !torch.float, !torch.int -> !torch.vtensor<[96],f32>
    %129 = torch.aten.sqrt %128 : !torch.vtensor<[96],f32> -> !torch.vtensor<[96],f32>
    %130 = torch.aten.reciprocal %129 : !torch.vtensor<[96],f32> -> !torch.vtensor<[96],f32>
    %int1_178 = torch.constant.int 1
    %131 = torch.aten.mul.Scalar %130, %int1_178 : !torch.vtensor<[96],f32>, !torch.int -> !torch.vtensor<[96],f32>
    %int0_179 = torch.constant.int 0
    %132 = torch.prim.ListConstruct %int0_179 : (!torch.int) -> !torch.list<int>
    %none_180 = torch.constant.none
    %none_181 = torch.constant.none
    %none_182 = torch.constant.none
    %false_183 = torch.constant.bool false
    %133 = torch.aten.new_zeros %125, %132, %none_180, %none_181, %none_182, %false_183 : !torch.vtensor<[1,96,56,56],f32>, !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[0],f32>
    %int0_184 = torch.constant.int 0
    %134 = torch.prim.ListConstruct %int0_184 : (!torch.int) -> !torch.list<int>
    %none_185 = torch.constant.none
    %none_186 = torch.constant.none
    %none_187 = torch.constant.none
    %false_188 = torch.constant.bool false
    %135 = torch.aten.new_zeros %125, %134, %none_185, %none_186, %none_187, %false_188 : !torch.vtensor<[1,96,56,56],f32>, !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[0],f32>
    %int-1_189 = torch.constant.int -1
    %136 = torch.aten.unsqueeze %126, %int-1_189 : !torch.vtensor<[96],f32>, !torch.int -> !torch.vtensor<[96,1],f32>
    %int-1_190 = torch.constant.int -1
    %137 = torch.aten.unsqueeze %136, %int-1_190 : !torch.vtensor<[96,1],f32>, !torch.int -> !torch.vtensor<[96,1,1],f32>
    %int-1_191 = torch.constant.int -1
    %138 = torch.aten.unsqueeze %131, %int-1_191 : !torch.vtensor<[96],f32>, !torch.int -> !torch.vtensor<[96,1],f32>
    %int-1_192 = torch.constant.int -1
    %139 = torch.aten.unsqueeze %138, %int-1_192 : !torch.vtensor<[96,1],f32>, !torch.int -> !torch.vtensor<[96,1,1],f32>
    %int1_193 = torch.constant.int 1
    %140 = torch.aten.sub.Tensor %125, %137, %int1_193 : !torch.vtensor<[1,96,56,56],f32>, !torch.vtensor<[96,1,1],f32>, !torch.int -> !torch.vtensor<[1,96,56,56],f32>
    %141 = torch.aten.mul.Tensor %140, %139 : !torch.vtensor<[1,96,56,56],f32>, !torch.vtensor<[96,1,1],f32> -> !torch.vtensor<[1,96,56,56],f32>
    %int-1_194 = torch.constant.int -1
    %142 = torch.aten.unsqueeze %arg24, %int-1_194 : !torch.vtensor<[96],f32>, !torch.int -> !torch.vtensor<[96,1],f32>
    %int-1_195 = torch.constant.int -1
    %143 = torch.aten.unsqueeze %142, %int-1_195 : !torch.vtensor<[96,1],f32>, !torch.int -> !torch.vtensor<[96,1,1],f32>
    %144 = torch.aten.mul.Tensor %141, %143 : !torch.vtensor<[1,96,56,56],f32>, !torch.vtensor<[96,1,1],f32> -> !torch.vtensor<[1,96,56,56],f32>
    %int-1_196 = torch.constant.int -1
    %145 = torch.aten.unsqueeze %arg25, %int-1_196 : !torch.vtensor<[96],f32>, !torch.int -> !torch.vtensor<[96,1],f32>
    %int-1_197 = torch.constant.int -1
    %146 = torch.aten.unsqueeze %145, %int-1_197 : !torch.vtensor<[96,1],f32>, !torch.int -> !torch.vtensor<[96,1,1],f32>
    %int1_198 = torch.constant.int 1
    %147 = torch.aten.add.Tensor %144, %146, %int1_198 : !torch.vtensor<[1,96,56,56],f32>, !torch.vtensor<[96,1,1],f32>, !torch.int -> !torch.vtensor<[1,96,56,56],f32>
    %float0.000000e00_199 = torch.constant.float 0.000000e+00
    %float6.000000e00_200 = torch.constant.float 6.000000e+00
    %148 = torch.aten.hardtanh %147, %float0.000000e00_199, %float6.000000e00_200 : !torch.vtensor<[1,96,56,56],f32>, !torch.float, !torch.float -> !torch.vtensor<[1,96,56,56],f32>
    %int0_201 = torch.constant.int 0
    %int0_202 = torch.constant.int 0
    %int0_203 = torch.constant.int 0
    %int0_204 = torch.constant.int 0
    %149 = torch.prim.ListConstruct %int0_201, %int0_202, %int0_203, %int0_204 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %float0.000000e00_205 = torch.constant.float 0.000000e+00
    %150 = torch.aten.constant_pad_nd %148, %149, %float0.000000e00_205 : !torch.vtensor<[1,96,56,56],f32>, !torch.list<int>, !torch.float -> !torch.vtensor<[1,96,56,56],f32>
    %none_206 = torch.constant.none
    %int1_207 = torch.constant.int 1
    %int1_208 = torch.constant.int 1
    %151 = torch.prim.ListConstruct %int1_207, %int1_208 : (!torch.int, !torch.int) -> !torch.list<int>
    %int0_209 = torch.constant.int 0
    %int0_210 = torch.constant.int 0
    %152 = torch.prim.ListConstruct %int0_209, %int0_210 : (!torch.int, !torch.int) -> !torch.list<int>
    %int1_211 = torch.constant.int 1
    %int1_212 = torch.constant.int 1
    %153 = torch.prim.ListConstruct %int1_211, %int1_212 : (!torch.int, !torch.int) -> !torch.list<int>
    %false_213 = torch.constant.bool false
    %int0_214 = torch.constant.int 0
    %int0_215 = torch.constant.int 0
    %154 = torch.prim.ListConstruct %int0_214, %int0_215 : (!torch.int, !torch.int) -> !torch.list<int>
    %int1_216 = torch.constant.int 1
    %155 = torch.aten.convolution %150, %arg26, %none_206, %151, %152, %153, %false_213, %154, %int1_216 : !torch.vtensor<[1,96,56,56],f32>, !torch.vtensor<[24,96,1,1],f32>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[1,24,56,56],f32>
    %int6_217 = torch.constant.int 6
    %156 = torch.prims.convert_element_type %arg27, %int6_217 : !torch.vtensor<[24],f32>, !torch.int -> !torch.vtensor<[24],f32>
    %int6_218 = torch.constant.int 6
    %157 = torch.prims.convert_element_type %arg28, %int6_218 : !torch.vtensor<[24],f32>, !torch.int -> !torch.vtensor<[24],f32>
    %float1.000000e-03_219 = torch.constant.float 1.000000e-03
    %int1_220 = torch.constant.int 1
    %158 = torch.aten.add.Scalar %157, %float1.000000e-03_219, %int1_220 : !torch.vtensor<[24],f32>, !torch.float, !torch.int -> !torch.vtensor<[24],f32>
    %159 = torch.aten.sqrt %158 : !torch.vtensor<[24],f32> -> !torch.vtensor<[24],f32>
    %160 = torch.aten.reciprocal %159 : !torch.vtensor<[24],f32> -> !torch.vtensor<[24],f32>
    %int1_221 = torch.constant.int 1
    %161 = torch.aten.mul.Scalar %160, %int1_221 : !torch.vtensor<[24],f32>, !torch.int -> !torch.vtensor<[24],f32>
    %int0_222 = torch.constant.int 0
    %162 = torch.prim.ListConstruct %int0_222 : (!torch.int) -> !torch.list<int>
    %none_223 = torch.constant.none
    %none_224 = torch.constant.none
    %none_225 = torch.constant.none
    %false_226 = torch.constant.bool false
    %163 = torch.aten.new_zeros %155, %162, %none_223, %none_224, %none_225, %false_226 : !torch.vtensor<[1,24,56,56],f32>, !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[0],f32>
    %int0_227 = torch.constant.int 0
    %164 = torch.prim.ListConstruct %int0_227 : (!torch.int) -> !torch.list<int>
    %none_228 = torch.constant.none
    %none_229 = torch.constant.none
    %none_230 = torch.constant.none
    %false_231 = torch.constant.bool false
    %165 = torch.aten.new_zeros %155, %164, %none_228, %none_229, %none_230, %false_231 : !torch.vtensor<[1,24,56,56],f32>, !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[0],f32>
    %int-1_232 = torch.constant.int -1
    %166 = torch.aten.unsqueeze %156, %int-1_232 : !torch.vtensor<[24],f32>, !torch.int -> !torch.vtensor<[24,1],f32>
    %int-1_233 = torch.constant.int -1
    %167 = torch.aten.unsqueeze %166, %int-1_233 : !torch.vtensor<[24,1],f32>, !torch.int -> !torch.vtensor<[24,1,1],f32>
    %int-1_234 = torch.constant.int -1
    %168 = torch.aten.unsqueeze %161, %int-1_234 : !torch.vtensor<[24],f32>, !torch.int -> !torch.vtensor<[24,1],f32>
    %int-1_235 = torch.constant.int -1
    %169 = torch.aten.unsqueeze %168, %int-1_235 : !torch.vtensor<[24,1],f32>, !torch.int -> !torch.vtensor<[24,1,1],f32>
    %int1_236 = torch.constant.int 1
    %170 = torch.aten.sub.Tensor %155, %167, %int1_236 : !torch.vtensor<[1,24,56,56],f32>, !torch.vtensor<[24,1,1],f32>, !torch.int -> !torch.vtensor<[1,24,56,56],f32>
    %171 = torch.aten.mul.Tensor %170, %169 : !torch.vtensor<[1,24,56,56],f32>, !torch.vtensor<[24,1,1],f32> -> !torch.vtensor<[1,24,56,56],f32>
    %int-1_237 = torch.constant.int -1
    %172 = torch.aten.unsqueeze %arg29, %int-1_237 : !torch.vtensor<[24],f32>, !torch.int -> !torch.vtensor<[24,1],f32>
    %int-1_238 = torch.constant.int -1
    %173 = torch.aten.unsqueeze %172, %int-1_238 : !torch.vtensor<[24,1],f32>, !torch.int -> !torch.vtensor<[24,1,1],f32>
    %174 = torch.aten.mul.Tensor %171, %173 : !torch.vtensor<[1,24,56,56],f32>, !torch.vtensor<[24,1,1],f32> -> !torch.vtensor<[1,24,56,56],f32>
    %int-1_239 = torch.constant.int -1
    %175 = torch.aten.unsqueeze %arg30, %int-1_239 : !torch.vtensor<[24],f32>, !torch.int -> !torch.vtensor<[24,1],f32>
    %int-1_240 = torch.constant.int -1
    %176 = torch.aten.unsqueeze %175, %int-1_240 : !torch.vtensor<[24,1],f32>, !torch.int -> !torch.vtensor<[24,1,1],f32>
    %int1_241 = torch.constant.int 1
    %177 = torch.aten.add.Tensor %174, %176, %int1_241 : !torch.vtensor<[1,24,56,56],f32>, !torch.vtensor<[24,1,1],f32>, !torch.int -> !torch.vtensor<[1,24,56,56],f32>
    %int0_242 = torch.constant.int 0
    %int0_243 = torch.constant.int 0
    %int0_244 = torch.constant.int 0
    %int0_245 = torch.constant.int 0
    %178 = torch.prim.ListConstruct %int0_242, %int0_243, %int0_244, %int0_245 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %float0.000000e00_246 = torch.constant.float 0.000000e+00
    %179 = torch.aten.constant_pad_nd %177, %178, %float0.000000e00_246 : !torch.vtensor<[1,24,56,56],f32>, !torch.list<int>, !torch.float -> !torch.vtensor<[1,24,56,56],f32>
    %none_247 = torch.constant.none
    %int1_248 = torch.constant.int 1
    %int1_249 = torch.constant.int 1
    %180 = torch.prim.ListConstruct %int1_248, %int1_249 : (!torch.int, !torch.int) -> !torch.list<int>
    %int0_250 = torch.constant.int 0
    %int0_251 = torch.constant.int 0
    %181 = torch.prim.ListConstruct %int0_250, %int0_251 : (!torch.int, !torch.int) -> !torch.list<int>
    %int1_252 = torch.constant.int 1
    %int1_253 = torch.constant.int 1
    %182 = torch.prim.ListConstruct %int1_252, %int1_253 : (!torch.int, !torch.int) -> !torch.list<int>
    %false_254 = torch.constant.bool false
    %int0_255 = torch.constant.int 0
    %int0_256 = torch.constant.int 0
    %183 = torch.prim.ListConstruct %int0_255, %int0_256 : (!torch.int, !torch.int) -> !torch.list<int>
    %int1_257 = torch.constant.int 1
    %184 = torch.aten.convolution %179, %arg31, %none_247, %180, %181, %182, %false_254, %183, %int1_257 : !torch.vtensor<[1,24,56,56],f32>, !torch.vtensor<[144,24,1,1],f32>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[1,144,56,56],f32>
    %int6_258 = torch.constant.int 6
    %185 = torch.prims.convert_element_type %arg32, %int6_258 : !torch.vtensor<[144],f32>, !torch.int -> !torch.vtensor<[144],f32>
    %int6_259 = torch.constant.int 6
    %186 = torch.prims.convert_element_type %arg33, %int6_259 : !torch.vtensor<[144],f32>, !torch.int -> !torch.vtensor<[144],f32>
    %float1.000000e-03_260 = torch.constant.float 1.000000e-03
    %int1_261 = torch.constant.int 1
    %187 = torch.aten.add.Scalar %186, %float1.000000e-03_260, %int1_261 : !torch.vtensor<[144],f32>, !torch.float, !torch.int -> !torch.vtensor<[144],f32>
    %188 = torch.aten.sqrt %187 : !torch.vtensor<[144],f32> -> !torch.vtensor<[144],f32>
    %189 = torch.aten.reciprocal %188 : !torch.vtensor<[144],f32> -> !torch.vtensor<[144],f32>
    %int1_262 = torch.constant.int 1
    %190 = torch.aten.mul.Scalar %189, %int1_262 : !torch.vtensor<[144],f32>, !torch.int -> !torch.vtensor<[144],f32>
    %int0_263 = torch.constant.int 0
    %191 = torch.prim.ListConstruct %int0_263 : (!torch.int) -> !torch.list<int>
    %none_264 = torch.constant.none
    %none_265 = torch.constant.none
    %none_266 = torch.constant.none
    %false_267 = torch.constant.bool false
    %192 = torch.aten.new_zeros %184, %191, %none_264, %none_265, %none_266, %false_267 : !torch.vtensor<[1,144,56,56],f32>, !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[0],f32>
    %int0_268 = torch.constant.int 0
    %193 = torch.prim.ListConstruct %int0_268 : (!torch.int) -> !torch.list<int>
    %none_269 = torch.constant.none
    %none_270 = torch.constant.none
    %none_271 = torch.constant.none
    %false_272 = torch.constant.bool false
    %194 = torch.aten.new_zeros %184, %193, %none_269, %none_270, %none_271, %false_272 : !torch.vtensor<[1,144,56,56],f32>, !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[0],f32>
    %int-1_273 = torch.constant.int -1
    %195 = torch.aten.unsqueeze %185, %int-1_273 : !torch.vtensor<[144],f32>, !torch.int -> !torch.vtensor<[144,1],f32>
    %int-1_274 = torch.constant.int -1
    %196 = torch.aten.unsqueeze %195, %int-1_274 : !torch.vtensor<[144,1],f32>, !torch.int -> !torch.vtensor<[144,1,1],f32>
    %int-1_275 = torch.constant.int -1
    %197 = torch.aten.unsqueeze %190, %int-1_275 : !torch.vtensor<[144],f32>, !torch.int -> !torch.vtensor<[144,1],f32>
    %int-1_276 = torch.constant.int -1
    %198 = torch.aten.unsqueeze %197, %int-1_276 : !torch.vtensor<[144,1],f32>, !torch.int -> !torch.vtensor<[144,1,1],f32>
    %int1_277 = torch.constant.int 1
    %199 = torch.aten.sub.Tensor %184, %196, %int1_277 : !torch.vtensor<[1,144,56,56],f32>, !torch.vtensor<[144,1,1],f32>, !torch.int -> !torch.vtensor<[1,144,56,56],f32>
    %200 = torch.aten.mul.Tensor %199, %198 : !torch.vtensor<[1,144,56,56],f32>, !torch.vtensor<[144,1,1],f32> -> !torch.vtensor<[1,144,56,56],f32>
    %int-1_278 = torch.constant.int -1
    %201 = torch.aten.unsqueeze %arg34, %int-1_278 : !torch.vtensor<[144],f32>, !torch.int -> !torch.vtensor<[144,1],f32>
    %int-1_279 = torch.constant.int -1
    %202 = torch.aten.unsqueeze %201, %int-1_279 : !torch.vtensor<[144,1],f32>, !torch.int -> !torch.vtensor<[144,1,1],f32>
    %203 = torch.aten.mul.Tensor %200, %202 : !torch.vtensor<[1,144,56,56],f32>, !torch.vtensor<[144,1,1],f32> -> !torch.vtensor<[1,144,56,56],f32>
    %int-1_280 = torch.constant.int -1
    %204 = torch.aten.unsqueeze %arg35, %int-1_280 : !torch.vtensor<[144],f32>, !torch.int -> !torch.vtensor<[144,1],f32>
    %int-1_281 = torch.constant.int -1
    %205 = torch.aten.unsqueeze %204, %int-1_281 : !torch.vtensor<[144,1],f32>, !torch.int -> !torch.vtensor<[144,1,1],f32>
    %int1_282 = torch.constant.int 1
    %206 = torch.aten.add.Tensor %203, %205, %int1_282 : !torch.vtensor<[1,144,56,56],f32>, !torch.vtensor<[144,1,1],f32>, !torch.int -> !torch.vtensor<[1,144,56,56],f32>
    %float0.000000e00_283 = torch.constant.float 0.000000e+00
    %float6.000000e00_284 = torch.constant.float 6.000000e+00
    %207 = torch.aten.hardtanh %206, %float0.000000e00_283, %float6.000000e00_284 : !torch.vtensor<[1,144,56,56],f32>, !torch.float, !torch.float -> !torch.vtensor<[1,144,56,56],f32>
    %int1_285 = torch.constant.int 1
    %int1_286 = torch.constant.int 1
    %int1_287 = torch.constant.int 1
    %int1_288 = torch.constant.int 1
    %208 = torch.prim.ListConstruct %int1_285, %int1_286, %int1_287, %int1_288 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %float0.000000e00_289 = torch.constant.float 0.000000e+00
    %209 = torch.aten.constant_pad_nd %207, %208, %float0.000000e00_289 : !torch.vtensor<[1,144,56,56],f32>, !torch.list<int>, !torch.float -> !torch.vtensor<[1,144,58,58],f32>
    %none_290 = torch.constant.none
    %int1_291 = torch.constant.int 1
    %int1_292 = torch.constant.int 1
    %210 = torch.prim.ListConstruct %int1_291, %int1_292 : (!torch.int, !torch.int) -> !torch.list<int>
    %int0_293 = torch.constant.int 0
    %int0_294 = torch.constant.int 0
    %211 = torch.prim.ListConstruct %int0_293, %int0_294 : (!torch.int, !torch.int) -> !torch.list<int>
    %int1_295 = torch.constant.int 1
    %int1_296 = torch.constant.int 1
    %212 = torch.prim.ListConstruct %int1_295, %int1_296 : (!torch.int, !torch.int) -> !torch.list<int>
    %false_297 = torch.constant.bool false
    %int0_298 = torch.constant.int 0
    %int0_299 = torch.constant.int 0
    %213 = torch.prim.ListConstruct %int0_298, %int0_299 : (!torch.int, !torch.int) -> !torch.list<int>
    %int144 = torch.constant.int 144
    %214 = torch.aten.convolution %209, %arg36, %none_290, %210, %211, %212, %false_297, %213, %int144 : !torch.vtensor<[1,144,58,58],f32>, !torch.vtensor<[144,1,3,3],f32>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[1,144,56,56],f32>
    %int6_300 = torch.constant.int 6
    %215 = torch.prims.convert_element_type %arg37, %int6_300 : !torch.vtensor<[144],f32>, !torch.int -> !torch.vtensor<[144],f32>
    %int6_301 = torch.constant.int 6
    %216 = torch.prims.convert_element_type %arg38, %int6_301 : !torch.vtensor<[144],f32>, !torch.int -> !torch.vtensor<[144],f32>
    %float1.000000e-03_302 = torch.constant.float 1.000000e-03
    %int1_303 = torch.constant.int 1
    %217 = torch.aten.add.Scalar %216, %float1.000000e-03_302, %int1_303 : !torch.vtensor<[144],f32>, !torch.float, !torch.int -> !torch.vtensor<[144],f32>
    %218 = torch.aten.sqrt %217 : !torch.vtensor<[144],f32> -> !torch.vtensor<[144],f32>
    %219 = torch.aten.reciprocal %218 : !torch.vtensor<[144],f32> -> !torch.vtensor<[144],f32>
    %int1_304 = torch.constant.int 1
    %220 = torch.aten.mul.Scalar %219, %int1_304 : !torch.vtensor<[144],f32>, !torch.int -> !torch.vtensor<[144],f32>
    %int0_305 = torch.constant.int 0
    %221 = torch.prim.ListConstruct %int0_305 : (!torch.int) -> !torch.list<int>
    %none_306 = torch.constant.none
    %none_307 = torch.constant.none
    %none_308 = torch.constant.none
    %false_309 = torch.constant.bool false
    %222 = torch.aten.new_zeros %214, %221, %none_306, %none_307, %none_308, %false_309 : !torch.vtensor<[1,144,56,56],f32>, !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[0],f32>
    %int0_310 = torch.constant.int 0
    %223 = torch.prim.ListConstruct %int0_310 : (!torch.int) -> !torch.list<int>
    %none_311 = torch.constant.none
    %none_312 = torch.constant.none
    %none_313 = torch.constant.none
    %false_314 = torch.constant.bool false
    %224 = torch.aten.new_zeros %214, %223, %none_311, %none_312, %none_313, %false_314 : !torch.vtensor<[1,144,56,56],f32>, !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[0],f32>
    %int-1_315 = torch.constant.int -1
    %225 = torch.aten.unsqueeze %215, %int-1_315 : !torch.vtensor<[144],f32>, !torch.int -> !torch.vtensor<[144,1],f32>
    %int-1_316 = torch.constant.int -1
    %226 = torch.aten.unsqueeze %225, %int-1_316 : !torch.vtensor<[144,1],f32>, !torch.int -> !torch.vtensor<[144,1,1],f32>
    %int-1_317 = torch.constant.int -1
    %227 = torch.aten.unsqueeze %220, %int-1_317 : !torch.vtensor<[144],f32>, !torch.int -> !torch.vtensor<[144,1],f32>
    %int-1_318 = torch.constant.int -1
    %228 = torch.aten.unsqueeze %227, %int-1_318 : !torch.vtensor<[144,1],f32>, !torch.int -> !torch.vtensor<[144,1,1],f32>
    %int1_319 = torch.constant.int 1
    %229 = torch.aten.sub.Tensor %214, %226, %int1_319 : !torch.vtensor<[1,144,56,56],f32>, !torch.vtensor<[144,1,1],f32>, !torch.int -> !torch.vtensor<[1,144,56,56],f32>
    %230 = torch.aten.mul.Tensor %229, %228 : !torch.vtensor<[1,144,56,56],f32>, !torch.vtensor<[144,1,1],f32> -> !torch.vtensor<[1,144,56,56],f32>
    %int-1_320 = torch.constant.int -1
    %231 = torch.aten.unsqueeze %arg39, %int-1_320 : !torch.vtensor<[144],f32>, !torch.int -> !torch.vtensor<[144,1],f32>
    %int-1_321 = torch.constant.int -1
    %232 = torch.aten.unsqueeze %231, %int-1_321 : !torch.vtensor<[144,1],f32>, !torch.int -> !torch.vtensor<[144,1,1],f32>
    %233 = torch.aten.mul.Tensor %230, %232 : !torch.vtensor<[1,144,56,56],f32>, !torch.vtensor<[144,1,1],f32> -> !torch.vtensor<[1,144,56,56],f32>
    %int-1_322 = torch.constant.int -1
    %234 = torch.aten.unsqueeze %arg40, %int-1_322 : !torch.vtensor<[144],f32>, !torch.int -> !torch.vtensor<[144,1],f32>
    %int-1_323 = torch.constant.int -1
    %235 = torch.aten.unsqueeze %234, %int-1_323 : !torch.vtensor<[144,1],f32>, !torch.int -> !torch.vtensor<[144,1,1],f32>
    %int1_324 = torch.constant.int 1
    %236 = torch.aten.add.Tensor %233, %235, %int1_324 : !torch.vtensor<[1,144,56,56],f32>, !torch.vtensor<[144,1,1],f32>, !torch.int -> !torch.vtensor<[1,144,56,56],f32>
    %float0.000000e00_325 = torch.constant.float 0.000000e+00
    %float6.000000e00_326 = torch.constant.float 6.000000e+00
    %237 = torch.aten.hardtanh %236, %float0.000000e00_325, %float6.000000e00_326 : !torch.vtensor<[1,144,56,56],f32>, !torch.float, !torch.float -> !torch.vtensor<[1,144,56,56],f32>
    %int0_327 = torch.constant.int 0
    %int0_328 = torch.constant.int 0
    %int0_329 = torch.constant.int 0
    %int0_330 = torch.constant.int 0
    %238 = torch.prim.ListConstruct %int0_327, %int0_328, %int0_329, %int0_330 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %float0.000000e00_331 = torch.constant.float 0.000000e+00
    %239 = torch.aten.constant_pad_nd %237, %238, %float0.000000e00_331 : !torch.vtensor<[1,144,56,56],f32>, !torch.list<int>, !torch.float -> !torch.vtensor<[1,144,56,56],f32>
    %none_332 = torch.constant.none
    %int1_333 = torch.constant.int 1
    %int1_334 = torch.constant.int 1
    %240 = torch.prim.ListConstruct %int1_333, %int1_334 : (!torch.int, !torch.int) -> !torch.list<int>
    %int0_335 = torch.constant.int 0
    %int0_336 = torch.constant.int 0
    %241 = torch.prim.ListConstruct %int0_335, %int0_336 : (!torch.int, !torch.int) -> !torch.list<int>
    %int1_337 = torch.constant.int 1
    %int1_338 = torch.constant.int 1
    %242 = torch.prim.ListConstruct %int1_337, %int1_338 : (!torch.int, !torch.int) -> !torch.list<int>
    %false_339 = torch.constant.bool false
    %int0_340 = torch.constant.int 0
    %int0_341 = torch.constant.int 0
    %243 = torch.prim.ListConstruct %int0_340, %int0_341 : (!torch.int, !torch.int) -> !torch.list<int>
    %int1_342 = torch.constant.int 1
    %244 = torch.aten.convolution %239, %arg41, %none_332, %240, %241, %242, %false_339, %243, %int1_342 : !torch.vtensor<[1,144,56,56],f32>, !torch.vtensor<[24,144,1,1],f32>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[1,24,56,56],f32>
    %int6_343 = torch.constant.int 6
    %245 = torch.prims.convert_element_type %arg42, %int6_343 : !torch.vtensor<[24],f32>, !torch.int -> !torch.vtensor<[24],f32>
    %int6_344 = torch.constant.int 6
    %246 = torch.prims.convert_element_type %arg43, %int6_344 : !torch.vtensor<[24],f32>, !torch.int -> !torch.vtensor<[24],f32>
    %float1.000000e-03_345 = torch.constant.float 1.000000e-03
    %int1_346 = torch.constant.int 1
    %247 = torch.aten.add.Scalar %246, %float1.000000e-03_345, %int1_346 : !torch.vtensor<[24],f32>, !torch.float, !torch.int -> !torch.vtensor<[24],f32>
    %248 = torch.aten.sqrt %247 : !torch.vtensor<[24],f32> -> !torch.vtensor<[24],f32>
    %249 = torch.aten.reciprocal %248 : !torch.vtensor<[24],f32> -> !torch.vtensor<[24],f32>
    %int1_347 = torch.constant.int 1
    %250 = torch.aten.mul.Scalar %249, %int1_347 : !torch.vtensor<[24],f32>, !torch.int -> !torch.vtensor<[24],f32>
    %int0_348 = torch.constant.int 0
    %251 = torch.prim.ListConstruct %int0_348 : (!torch.int) -> !torch.list<int>
    %none_349 = torch.constant.none
    %none_350 = torch.constant.none
    %none_351 = torch.constant.none
    %false_352 = torch.constant.bool false
    %252 = torch.aten.new_zeros %244, %251, %none_349, %none_350, %none_351, %false_352 : !torch.vtensor<[1,24,56,56],f32>, !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[0],f32>
    %int0_353 = torch.constant.int 0
    %253 = torch.prim.ListConstruct %int0_353 : (!torch.int) -> !torch.list<int>
    %none_354 = torch.constant.none
    %none_355 = torch.constant.none
    %none_356 = torch.constant.none
    %false_357 = torch.constant.bool false
    %254 = torch.aten.new_zeros %244, %253, %none_354, %none_355, %none_356, %false_357 : !torch.vtensor<[1,24,56,56],f32>, !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[0],f32>
    %int-1_358 = torch.constant.int -1
    %255 = torch.aten.unsqueeze %245, %int-1_358 : !torch.vtensor<[24],f32>, !torch.int -> !torch.vtensor<[24,1],f32>
    %int-1_359 = torch.constant.int -1
    %256 = torch.aten.unsqueeze %255, %int-1_359 : !torch.vtensor<[24,1],f32>, !torch.int -> !torch.vtensor<[24,1,1],f32>
    %int-1_360 = torch.constant.int -1
    %257 = torch.aten.unsqueeze %250, %int-1_360 : !torch.vtensor<[24],f32>, !torch.int -> !torch.vtensor<[24,1],f32>
    %int-1_361 = torch.constant.int -1
    %258 = torch.aten.unsqueeze %257, %int-1_361 : !torch.vtensor<[24,1],f32>, !torch.int -> !torch.vtensor<[24,1,1],f32>
    %int1_362 = torch.constant.int 1
    %259 = torch.aten.sub.Tensor %244, %256, %int1_362 : !torch.vtensor<[1,24,56,56],f32>, !torch.vtensor<[24,1,1],f32>, !torch.int -> !torch.vtensor<[1,24,56,56],f32>
    %260 = torch.aten.mul.Tensor %259, %258 : !torch.vtensor<[1,24,56,56],f32>, !torch.vtensor<[24,1,1],f32> -> !torch.vtensor<[1,24,56,56],f32>
    %int-1_363 = torch.constant.int -1
    %261 = torch.aten.unsqueeze %arg44, %int-1_363 : !torch.vtensor<[24],f32>, !torch.int -> !torch.vtensor<[24,1],f32>
    %int-1_364 = torch.constant.int -1
    %262 = torch.aten.unsqueeze %261, %int-1_364 : !torch.vtensor<[24,1],f32>, !torch.int -> !torch.vtensor<[24,1,1],f32>
    %263 = torch.aten.mul.Tensor %260, %262 : !torch.vtensor<[1,24,56,56],f32>, !torch.vtensor<[24,1,1],f32> -> !torch.vtensor<[1,24,56,56],f32>
    %int-1_365 = torch.constant.int -1
    %264 = torch.aten.unsqueeze %arg45, %int-1_365 : !torch.vtensor<[24],f32>, !torch.int -> !torch.vtensor<[24,1],f32>
    %int-1_366 = torch.constant.int -1
    %265 = torch.aten.unsqueeze %264, %int-1_366 : !torch.vtensor<[24,1],f32>, !torch.int -> !torch.vtensor<[24,1,1],f32>
    %int1_367 = torch.constant.int 1
    %266 = torch.aten.add.Tensor %263, %265, %int1_367 : !torch.vtensor<[1,24,56,56],f32>, !torch.vtensor<[24,1,1],f32>, !torch.int -> !torch.vtensor<[1,24,56,56],f32>
    %int1_368 = torch.constant.int 1
    %267 = torch.aten.add.Tensor %177, %266, %int1_368 : !torch.vtensor<[1,24,56,56],f32>, !torch.vtensor<[1,24,56,56],f32>, !torch.int -> !torch.vtensor<[1,24,56,56],f32>
    %int0_369 = torch.constant.int 0
    %int0_370 = torch.constant.int 0
    %int0_371 = torch.constant.int 0
    %int0_372 = torch.constant.int 0
    %268 = torch.prim.ListConstruct %int0_369, %int0_370, %int0_371, %int0_372 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %float0.000000e00_373 = torch.constant.float 0.000000e+00
    %269 = torch.aten.constant_pad_nd %267, %268, %float0.000000e00_373 : !torch.vtensor<[1,24,56,56],f32>, !torch.list<int>, !torch.float -> !torch.vtensor<[1,24,56,56],f32>
    %none_374 = torch.constant.none
    %int1_375 = torch.constant.int 1
    %int1_376 = torch.constant.int 1
    %270 = torch.prim.ListConstruct %int1_375, %int1_376 : (!torch.int, !torch.int) -> !torch.list<int>
    %int0_377 = torch.constant.int 0
    %int0_378 = torch.constant.int 0
    %271 = torch.prim.ListConstruct %int0_377, %int0_378 : (!torch.int, !torch.int) -> !torch.list<int>
    %int1_379 = torch.constant.int 1
    %int1_380 = torch.constant.int 1
    %272 = torch.prim.ListConstruct %int1_379, %int1_380 : (!torch.int, !torch.int) -> !torch.list<int>
    %false_381 = torch.constant.bool false
    %int0_382 = torch.constant.int 0
    %int0_383 = torch.constant.int 0
    %273 = torch.prim.ListConstruct %int0_382, %int0_383 : (!torch.int, !torch.int) -> !torch.list<int>
    %int1_384 = torch.constant.int 1
    %274 = torch.aten.convolution %269, %arg46, %none_374, %270, %271, %272, %false_381, %273, %int1_384 : !torch.vtensor<[1,24,56,56],f32>, !torch.vtensor<[144,24,1,1],f32>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[1,144,56,56],f32>
    %int6_385 = torch.constant.int 6
    %275 = torch.prims.convert_element_type %arg47, %int6_385 : !torch.vtensor<[144],f32>, !torch.int -> !torch.vtensor<[144],f32>
    %int6_386 = torch.constant.int 6
    %276 = torch.prims.convert_element_type %arg48, %int6_386 : !torch.vtensor<[144],f32>, !torch.int -> !torch.vtensor<[144],f32>
    %float1.000000e-03_387 = torch.constant.float 1.000000e-03
    %int1_388 = torch.constant.int 1
    %277 = torch.aten.add.Scalar %276, %float1.000000e-03_387, %int1_388 : !torch.vtensor<[144],f32>, !torch.float, !torch.int -> !torch.vtensor<[144],f32>
    %278 = torch.aten.sqrt %277 : !torch.vtensor<[144],f32> -> !torch.vtensor<[144],f32>
    %279 = torch.aten.reciprocal %278 : !torch.vtensor<[144],f32> -> !torch.vtensor<[144],f32>
    %int1_389 = torch.constant.int 1
    %280 = torch.aten.mul.Scalar %279, %int1_389 : !torch.vtensor<[144],f32>, !torch.int -> !torch.vtensor<[144],f32>
    %int0_390 = torch.constant.int 0
    %281 = torch.prim.ListConstruct %int0_390 : (!torch.int) -> !torch.list<int>
    %none_391 = torch.constant.none
    %none_392 = torch.constant.none
    %none_393 = torch.constant.none
    %false_394 = torch.constant.bool false
    %282 = torch.aten.new_zeros %274, %281, %none_391, %none_392, %none_393, %false_394 : !torch.vtensor<[1,144,56,56],f32>, !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[0],f32>
    %int0_395 = torch.constant.int 0
    %283 = torch.prim.ListConstruct %int0_395 : (!torch.int) -> !torch.list<int>
    %none_396 = torch.constant.none
    %none_397 = torch.constant.none
    %none_398 = torch.constant.none
    %false_399 = torch.constant.bool false
    %284 = torch.aten.new_zeros %274, %283, %none_396, %none_397, %none_398, %false_399 : !torch.vtensor<[1,144,56,56],f32>, !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[0],f32>
    %int-1_400 = torch.constant.int -1
    %285 = torch.aten.unsqueeze %275, %int-1_400 : !torch.vtensor<[144],f32>, !torch.int -> !torch.vtensor<[144,1],f32>
    %int-1_401 = torch.constant.int -1
    %286 = torch.aten.unsqueeze %285, %int-1_401 : !torch.vtensor<[144,1],f32>, !torch.int -> !torch.vtensor<[144,1,1],f32>
    %int-1_402 = torch.constant.int -1
    %287 = torch.aten.unsqueeze %280, %int-1_402 : !torch.vtensor<[144],f32>, !torch.int -> !torch.vtensor<[144,1],f32>
    %int-1_403 = torch.constant.int -1
    %288 = torch.aten.unsqueeze %287, %int-1_403 : !torch.vtensor<[144,1],f32>, !torch.int -> !torch.vtensor<[144,1,1],f32>
    %int1_404 = torch.constant.int 1
    %289 = torch.aten.sub.Tensor %274, %286, %int1_404 : !torch.vtensor<[1,144,56,56],f32>, !torch.vtensor<[144,1,1],f32>, !torch.int -> !torch.vtensor<[1,144,56,56],f32>
    %290 = torch.aten.mul.Tensor %289, %288 : !torch.vtensor<[1,144,56,56],f32>, !torch.vtensor<[144,1,1],f32> -> !torch.vtensor<[1,144,56,56],f32>
    %int-1_405 = torch.constant.int -1
    %291 = torch.aten.unsqueeze %arg49, %int-1_405 : !torch.vtensor<[144],f32>, !torch.int -> !torch.vtensor<[144,1],f32>
    %int-1_406 = torch.constant.int -1
    %292 = torch.aten.unsqueeze %291, %int-1_406 : !torch.vtensor<[144,1],f32>, !torch.int -> !torch.vtensor<[144,1,1],f32>
    %293 = torch.aten.mul.Tensor %290, %292 : !torch.vtensor<[1,144,56,56],f32>, !torch.vtensor<[144,1,1],f32> -> !torch.vtensor<[1,144,56,56],f32>
    %int-1_407 = torch.constant.int -1
    %294 = torch.aten.unsqueeze %arg50, %int-1_407 : !torch.vtensor<[144],f32>, !torch.int -> !torch.vtensor<[144,1],f32>
    %int-1_408 = torch.constant.int -1
    %295 = torch.aten.unsqueeze %294, %int-1_408 : !torch.vtensor<[144,1],f32>, !torch.int -> !torch.vtensor<[144,1,1],f32>
    %int1_409 = torch.constant.int 1
    %296 = torch.aten.add.Tensor %293, %295, %int1_409 : !torch.vtensor<[1,144,56,56],f32>, !torch.vtensor<[144,1,1],f32>, !torch.int -> !torch.vtensor<[1,144,56,56],f32>
    %float0.000000e00_410 = torch.constant.float 0.000000e+00
    %float6.000000e00_411 = torch.constant.float 6.000000e+00
    %297 = torch.aten.hardtanh %296, %float0.000000e00_410, %float6.000000e00_411 : !torch.vtensor<[1,144,56,56],f32>, !torch.float, !torch.float -> !torch.vtensor<[1,144,56,56],f32>
    %int0_412 = torch.constant.int 0
    %int1_413 = torch.constant.int 1
    %int0_414 = torch.constant.int 0
    %int1_415 = torch.constant.int 1
    %298 = torch.prim.ListConstruct %int0_412, %int1_413, %int0_414, %int1_415 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %float0.000000e00_416 = torch.constant.float 0.000000e+00
    %299 = torch.aten.constant_pad_nd %297, %298, %float0.000000e00_416 : !torch.vtensor<[1,144,56,56],f32>, !torch.list<int>, !torch.float -> !torch.vtensor<[1,144,57,57],f32>
    %none_417 = torch.constant.none
    %int2_418 = torch.constant.int 2
    %int2_419 = torch.constant.int 2
    %300 = torch.prim.ListConstruct %int2_418, %int2_419 : (!torch.int, !torch.int) -> !torch.list<int>
    %int0_420 = torch.constant.int 0
    %int0_421 = torch.constant.int 0
    %301 = torch.prim.ListConstruct %int0_420, %int0_421 : (!torch.int, !torch.int) -> !torch.list<int>
    %int1_422 = torch.constant.int 1
    %int1_423 = torch.constant.int 1
    %302 = torch.prim.ListConstruct %int1_422, %int1_423 : (!torch.int, !torch.int) -> !torch.list<int>
    %false_424 = torch.constant.bool false
    %int0_425 = torch.constant.int 0
    %int0_426 = torch.constant.int 0
    %303 = torch.prim.ListConstruct %int0_425, %int0_426 : (!torch.int, !torch.int) -> !torch.list<int>
    %int144_427 = torch.constant.int 144
    %304 = torch.aten.convolution %299, %arg51, %none_417, %300, %301, %302, %false_424, %303, %int144_427 : !torch.vtensor<[1,144,57,57],f32>, !torch.vtensor<[144,1,3,3],f32>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[1,144,28,28],f32>
    %int6_428 = torch.constant.int 6
    %305 = torch.prims.convert_element_type %arg52, %int6_428 : !torch.vtensor<[144],f32>, !torch.int -> !torch.vtensor<[144],f32>
    %int6_429 = torch.constant.int 6
    %306 = torch.prims.convert_element_type %arg53, %int6_429 : !torch.vtensor<[144],f32>, !torch.int -> !torch.vtensor<[144],f32>
    %float1.000000e-03_430 = torch.constant.float 1.000000e-03
    %int1_431 = torch.constant.int 1
    %307 = torch.aten.add.Scalar %306, %float1.000000e-03_430, %int1_431 : !torch.vtensor<[144],f32>, !torch.float, !torch.int -> !torch.vtensor<[144],f32>
    %308 = torch.aten.sqrt %307 : !torch.vtensor<[144],f32> -> !torch.vtensor<[144],f32>
    %309 = torch.aten.reciprocal %308 : !torch.vtensor<[144],f32> -> !torch.vtensor<[144],f32>
    %int1_432 = torch.constant.int 1
    %310 = torch.aten.mul.Scalar %309, %int1_432 : !torch.vtensor<[144],f32>, !torch.int -> !torch.vtensor<[144],f32>
    %int0_433 = torch.constant.int 0
    %311 = torch.prim.ListConstruct %int0_433 : (!torch.int) -> !torch.list<int>
    %none_434 = torch.constant.none
    %none_435 = torch.constant.none
    %none_436 = torch.constant.none
    %false_437 = torch.constant.bool false
    %312 = torch.aten.new_zeros %304, %311, %none_434, %none_435, %none_436, %false_437 : !torch.vtensor<[1,144,28,28],f32>, !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[0],f32>
    %int0_438 = torch.constant.int 0
    %313 = torch.prim.ListConstruct %int0_438 : (!torch.int) -> !torch.list<int>
    %none_439 = torch.constant.none
    %none_440 = torch.constant.none
    %none_441 = torch.constant.none
    %false_442 = torch.constant.bool false
    %314 = torch.aten.new_zeros %304, %313, %none_439, %none_440, %none_441, %false_442 : !torch.vtensor<[1,144,28,28],f32>, !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[0],f32>
    %int-1_443 = torch.constant.int -1
    %315 = torch.aten.unsqueeze %305, %int-1_443 : !torch.vtensor<[144],f32>, !torch.int -> !torch.vtensor<[144,1],f32>
    %int-1_444 = torch.constant.int -1
    %316 = torch.aten.unsqueeze %315, %int-1_444 : !torch.vtensor<[144,1],f32>, !torch.int -> !torch.vtensor<[144,1,1],f32>
    %int-1_445 = torch.constant.int -1
    %317 = torch.aten.unsqueeze %310, %int-1_445 : !torch.vtensor<[144],f32>, !torch.int -> !torch.vtensor<[144,1],f32>
    %int-1_446 = torch.constant.int -1
    %318 = torch.aten.unsqueeze %317, %int-1_446 : !torch.vtensor<[144,1],f32>, !torch.int -> !torch.vtensor<[144,1,1],f32>
    %int1_447 = torch.constant.int 1
    %319 = torch.aten.sub.Tensor %304, %316, %int1_447 : !torch.vtensor<[1,144,28,28],f32>, !torch.vtensor<[144,1,1],f32>, !torch.int -> !torch.vtensor<[1,144,28,28],f32>
    %320 = torch.aten.mul.Tensor %319, %318 : !torch.vtensor<[1,144,28,28],f32>, !torch.vtensor<[144,1,1],f32> -> !torch.vtensor<[1,144,28,28],f32>
    %int-1_448 = torch.constant.int -1
    %321 = torch.aten.unsqueeze %arg54, %int-1_448 : !torch.vtensor<[144],f32>, !torch.int -> !torch.vtensor<[144,1],f32>
    %int-1_449 = torch.constant.int -1
    %322 = torch.aten.unsqueeze %321, %int-1_449 : !torch.vtensor<[144,1],f32>, !torch.int -> !torch.vtensor<[144,1,1],f32>
    %323 = torch.aten.mul.Tensor %320, %322 : !torch.vtensor<[1,144,28,28],f32>, !torch.vtensor<[144,1,1],f32> -> !torch.vtensor<[1,144,28,28],f32>
    %int-1_450 = torch.constant.int -1
    %324 = torch.aten.unsqueeze %arg55, %int-1_450 : !torch.vtensor<[144],f32>, !torch.int -> !torch.vtensor<[144,1],f32>
    %int-1_451 = torch.constant.int -1
    %325 = torch.aten.unsqueeze %324, %int-1_451 : !torch.vtensor<[144,1],f32>, !torch.int -> !torch.vtensor<[144,1,1],f32>
    %int1_452 = torch.constant.int 1
    %326 = torch.aten.add.Tensor %323, %325, %int1_452 : !torch.vtensor<[1,144,28,28],f32>, !torch.vtensor<[144,1,1],f32>, !torch.int -> !torch.vtensor<[1,144,28,28],f32>
    %float0.000000e00_453 = torch.constant.float 0.000000e+00
    %float6.000000e00_454 = torch.constant.float 6.000000e+00
    %327 = torch.aten.hardtanh %326, %float0.000000e00_453, %float6.000000e00_454 : !torch.vtensor<[1,144,28,28],f32>, !torch.float, !torch.float -> !torch.vtensor<[1,144,28,28],f32>
    %int0_455 = torch.constant.int 0
    %int0_456 = torch.constant.int 0
    %int0_457 = torch.constant.int 0
    %int0_458 = torch.constant.int 0
    %328 = torch.prim.ListConstruct %int0_455, %int0_456, %int0_457, %int0_458 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %float0.000000e00_459 = torch.constant.float 0.000000e+00
    %329 = torch.aten.constant_pad_nd %327, %328, %float0.000000e00_459 : !torch.vtensor<[1,144,28,28],f32>, !torch.list<int>, !torch.float -> !torch.vtensor<[1,144,28,28],f32>
    %none_460 = torch.constant.none
    %int1_461 = torch.constant.int 1
    %int1_462 = torch.constant.int 1
    %330 = torch.prim.ListConstruct %int1_461, %int1_462 : (!torch.int, !torch.int) -> !torch.list<int>
    %int0_463 = torch.constant.int 0
    %int0_464 = torch.constant.int 0
    %331 = torch.prim.ListConstruct %int0_463, %int0_464 : (!torch.int, !torch.int) -> !torch.list<int>
    %int1_465 = torch.constant.int 1
    %int1_466 = torch.constant.int 1
    %332 = torch.prim.ListConstruct %int1_465, %int1_466 : (!torch.int, !torch.int) -> !torch.list<int>
    %false_467 = torch.constant.bool false
    %int0_468 = torch.constant.int 0
    %int0_469 = torch.constant.int 0
    %333 = torch.prim.ListConstruct %int0_468, %int0_469 : (!torch.int, !torch.int) -> !torch.list<int>
    %int1_470 = torch.constant.int 1
    %334 = torch.aten.convolution %329, %arg56, %none_460, %330, %331, %332, %false_467, %333, %int1_470 : !torch.vtensor<[1,144,28,28],f32>, !torch.vtensor<[32,144,1,1],f32>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[1,32,28,28],f32>
    %int6_471 = torch.constant.int 6
    %335 = torch.prims.convert_element_type %arg57, %int6_471 : !torch.vtensor<[32],f32>, !torch.int -> !torch.vtensor<[32],f32>
    %int6_472 = torch.constant.int 6
    %336 = torch.prims.convert_element_type %arg58, %int6_472 : !torch.vtensor<[32],f32>, !torch.int -> !torch.vtensor<[32],f32>
    %float1.000000e-03_473 = torch.constant.float 1.000000e-03
    %int1_474 = torch.constant.int 1
    %337 = torch.aten.add.Scalar %336, %float1.000000e-03_473, %int1_474 : !torch.vtensor<[32],f32>, !torch.float, !torch.int -> !torch.vtensor<[32],f32>
    %338 = torch.aten.sqrt %337 : !torch.vtensor<[32],f32> -> !torch.vtensor<[32],f32>
    %339 = torch.aten.reciprocal %338 : !torch.vtensor<[32],f32> -> !torch.vtensor<[32],f32>
    %int1_475 = torch.constant.int 1
    %340 = torch.aten.mul.Scalar %339, %int1_475 : !torch.vtensor<[32],f32>, !torch.int -> !torch.vtensor<[32],f32>
    %int0_476 = torch.constant.int 0
    %341 = torch.prim.ListConstruct %int0_476 : (!torch.int) -> !torch.list<int>
    %none_477 = torch.constant.none
    %none_478 = torch.constant.none
    %none_479 = torch.constant.none
    %false_480 = torch.constant.bool false
    %342 = torch.aten.new_zeros %334, %341, %none_477, %none_478, %none_479, %false_480 : !torch.vtensor<[1,32,28,28],f32>, !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[0],f32>
    %int0_481 = torch.constant.int 0
    %343 = torch.prim.ListConstruct %int0_481 : (!torch.int) -> !torch.list<int>
    %none_482 = torch.constant.none
    %none_483 = torch.constant.none
    %none_484 = torch.constant.none
    %false_485 = torch.constant.bool false
    %344 = torch.aten.new_zeros %334, %343, %none_482, %none_483, %none_484, %false_485 : !torch.vtensor<[1,32,28,28],f32>, !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[0],f32>
    %int-1_486 = torch.constant.int -1
    %345 = torch.aten.unsqueeze %335, %int-1_486 : !torch.vtensor<[32],f32>, !torch.int -> !torch.vtensor<[32,1],f32>
    %int-1_487 = torch.constant.int -1
    %346 = torch.aten.unsqueeze %345, %int-1_487 : !torch.vtensor<[32,1],f32>, !torch.int -> !torch.vtensor<[32,1,1],f32>
    %int-1_488 = torch.constant.int -1
    %347 = torch.aten.unsqueeze %340, %int-1_488 : !torch.vtensor<[32],f32>, !torch.int -> !torch.vtensor<[32,1],f32>
    %int-1_489 = torch.constant.int -1
    %348 = torch.aten.unsqueeze %347, %int-1_489 : !torch.vtensor<[32,1],f32>, !torch.int -> !torch.vtensor<[32,1,1],f32>
    %int1_490 = torch.constant.int 1
    %349 = torch.aten.sub.Tensor %334, %346, %int1_490 : !torch.vtensor<[1,32,28,28],f32>, !torch.vtensor<[32,1,1],f32>, !torch.int -> !torch.vtensor<[1,32,28,28],f32>
    %350 = torch.aten.mul.Tensor %349, %348 : !torch.vtensor<[1,32,28,28],f32>, !torch.vtensor<[32,1,1],f32> -> !torch.vtensor<[1,32,28,28],f32>
    %int-1_491 = torch.constant.int -1
    %351 = torch.aten.unsqueeze %arg59, %int-1_491 : !torch.vtensor<[32],f32>, !torch.int -> !torch.vtensor<[32,1],f32>
    %int-1_492 = torch.constant.int -1
    %352 = torch.aten.unsqueeze %351, %int-1_492 : !torch.vtensor<[32,1],f32>, !torch.int -> !torch.vtensor<[32,1,1],f32>
    %353 = torch.aten.mul.Tensor %350, %352 : !torch.vtensor<[1,32,28,28],f32>, !torch.vtensor<[32,1,1],f32> -> !torch.vtensor<[1,32,28,28],f32>
    %int-1_493 = torch.constant.int -1
    %354 = torch.aten.unsqueeze %arg60, %int-1_493 : !torch.vtensor<[32],f32>, !torch.int -> !torch.vtensor<[32,1],f32>
    %int-1_494 = torch.constant.int -1
    %355 = torch.aten.unsqueeze %354, %int-1_494 : !torch.vtensor<[32,1],f32>, !torch.int -> !torch.vtensor<[32,1,1],f32>
    %int1_495 = torch.constant.int 1
    %356 = torch.aten.add.Tensor %353, %355, %int1_495 : !torch.vtensor<[1,32,28,28],f32>, !torch.vtensor<[32,1,1],f32>, !torch.int -> !torch.vtensor<[1,32,28,28],f32>
    %int0_496 = torch.constant.int 0
    %int0_497 = torch.constant.int 0
    %int0_498 = torch.constant.int 0
    %int0_499 = torch.constant.int 0
    %357 = torch.prim.ListConstruct %int0_496, %int0_497, %int0_498, %int0_499 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %float0.000000e00_500 = torch.constant.float 0.000000e+00
    %358 = torch.aten.constant_pad_nd %356, %357, %float0.000000e00_500 : !torch.vtensor<[1,32,28,28],f32>, !torch.list<int>, !torch.float -> !torch.vtensor<[1,32,28,28],f32>
    %none_501 = torch.constant.none
    %int1_502 = torch.constant.int 1
    %int1_503 = torch.constant.int 1
    %359 = torch.prim.ListConstruct %int1_502, %int1_503 : (!torch.int, !torch.int) -> !torch.list<int>
    %int0_504 = torch.constant.int 0
    %int0_505 = torch.constant.int 0
    %360 = torch.prim.ListConstruct %int0_504, %int0_505 : (!torch.int, !torch.int) -> !torch.list<int>
    %int1_506 = torch.constant.int 1
    %int1_507 = torch.constant.int 1
    %361 = torch.prim.ListConstruct %int1_506, %int1_507 : (!torch.int, !torch.int) -> !torch.list<int>
    %false_508 = torch.constant.bool false
    %int0_509 = torch.constant.int 0
    %int0_510 = torch.constant.int 0
    %362 = torch.prim.ListConstruct %int0_509, %int0_510 : (!torch.int, !torch.int) -> !torch.list<int>
    %int1_511 = torch.constant.int 1
    %363 = torch.aten.convolution %358, %arg61, %none_501, %359, %360, %361, %false_508, %362, %int1_511 : !torch.vtensor<[1,32,28,28],f32>, !torch.vtensor<[192,32,1,1],f32>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[1,192,28,28],f32>
    %int6_512 = torch.constant.int 6
    %364 = torch.prims.convert_element_type %arg62, %int6_512 : !torch.vtensor<[192],f32>, !torch.int -> !torch.vtensor<[192],f32>
    %int6_513 = torch.constant.int 6
    %365 = torch.prims.convert_element_type %arg63, %int6_513 : !torch.vtensor<[192],f32>, !torch.int -> !torch.vtensor<[192],f32>
    %float1.000000e-03_514 = torch.constant.float 1.000000e-03
    %int1_515 = torch.constant.int 1
    %366 = torch.aten.add.Scalar %365, %float1.000000e-03_514, %int1_515 : !torch.vtensor<[192],f32>, !torch.float, !torch.int -> !torch.vtensor<[192],f32>
    %367 = torch.aten.sqrt %366 : !torch.vtensor<[192],f32> -> !torch.vtensor<[192],f32>
    %368 = torch.aten.reciprocal %367 : !torch.vtensor<[192],f32> -> !torch.vtensor<[192],f32>
    %int1_516 = torch.constant.int 1
    %369 = torch.aten.mul.Scalar %368, %int1_516 : !torch.vtensor<[192],f32>, !torch.int -> !torch.vtensor<[192],f32>
    %int0_517 = torch.constant.int 0
    %370 = torch.prim.ListConstruct %int0_517 : (!torch.int) -> !torch.list<int>
    %none_518 = torch.constant.none
    %none_519 = torch.constant.none
    %none_520 = torch.constant.none
    %false_521 = torch.constant.bool false
    %371 = torch.aten.new_zeros %363, %370, %none_518, %none_519, %none_520, %false_521 : !torch.vtensor<[1,192,28,28],f32>, !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[0],f32>
    %int0_522 = torch.constant.int 0
    %372 = torch.prim.ListConstruct %int0_522 : (!torch.int) -> !torch.list<int>
    %none_523 = torch.constant.none
    %none_524 = torch.constant.none
    %none_525 = torch.constant.none
    %false_526 = torch.constant.bool false
    %373 = torch.aten.new_zeros %363, %372, %none_523, %none_524, %none_525, %false_526 : !torch.vtensor<[1,192,28,28],f32>, !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[0],f32>
    %int-1_527 = torch.constant.int -1
    %374 = torch.aten.unsqueeze %364, %int-1_527 : !torch.vtensor<[192],f32>, !torch.int -> !torch.vtensor<[192,1],f32>
    %int-1_528 = torch.constant.int -1
    %375 = torch.aten.unsqueeze %374, %int-1_528 : !torch.vtensor<[192,1],f32>, !torch.int -> !torch.vtensor<[192,1,1],f32>
    %int-1_529 = torch.constant.int -1
    %376 = torch.aten.unsqueeze %369, %int-1_529 : !torch.vtensor<[192],f32>, !torch.int -> !torch.vtensor<[192,1],f32>
    %int-1_530 = torch.constant.int -1
    %377 = torch.aten.unsqueeze %376, %int-1_530 : !torch.vtensor<[192,1],f32>, !torch.int -> !torch.vtensor<[192,1,1],f32>
    %int1_531 = torch.constant.int 1
    %378 = torch.aten.sub.Tensor %363, %375, %int1_531 : !torch.vtensor<[1,192,28,28],f32>, !torch.vtensor<[192,1,1],f32>, !torch.int -> !torch.vtensor<[1,192,28,28],f32>
    %379 = torch.aten.mul.Tensor %378, %377 : !torch.vtensor<[1,192,28,28],f32>, !torch.vtensor<[192,1,1],f32> -> !torch.vtensor<[1,192,28,28],f32>
    %int-1_532 = torch.constant.int -1
    %380 = torch.aten.unsqueeze %arg64, %int-1_532 : !torch.vtensor<[192],f32>, !torch.int -> !torch.vtensor<[192,1],f32>
    %int-1_533 = torch.constant.int -1
    %381 = torch.aten.unsqueeze %380, %int-1_533 : !torch.vtensor<[192,1],f32>, !torch.int -> !torch.vtensor<[192,1,1],f32>
    %382 = torch.aten.mul.Tensor %379, %381 : !torch.vtensor<[1,192,28,28],f32>, !torch.vtensor<[192,1,1],f32> -> !torch.vtensor<[1,192,28,28],f32>
    %int-1_534 = torch.constant.int -1
    %383 = torch.aten.unsqueeze %arg65, %int-1_534 : !torch.vtensor<[192],f32>, !torch.int -> !torch.vtensor<[192,1],f32>
    %int-1_535 = torch.constant.int -1
    %384 = torch.aten.unsqueeze %383, %int-1_535 : !torch.vtensor<[192,1],f32>, !torch.int -> !torch.vtensor<[192,1,1],f32>
    %int1_536 = torch.constant.int 1
    %385 = torch.aten.add.Tensor %382, %384, %int1_536 : !torch.vtensor<[1,192,28,28],f32>, !torch.vtensor<[192,1,1],f32>, !torch.int -> !torch.vtensor<[1,192,28,28],f32>
    %float0.000000e00_537 = torch.constant.float 0.000000e+00
    %float6.000000e00_538 = torch.constant.float 6.000000e+00
    %386 = torch.aten.hardtanh %385, %float0.000000e00_537, %float6.000000e00_538 : !torch.vtensor<[1,192,28,28],f32>, !torch.float, !torch.float -> !torch.vtensor<[1,192,28,28],f32>
    %int1_539 = torch.constant.int 1
    %int1_540 = torch.constant.int 1
    %int1_541 = torch.constant.int 1
    %int1_542 = torch.constant.int 1
    %387 = torch.prim.ListConstruct %int1_539, %int1_540, %int1_541, %int1_542 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %float0.000000e00_543 = torch.constant.float 0.000000e+00
    %388 = torch.aten.constant_pad_nd %386, %387, %float0.000000e00_543 : !torch.vtensor<[1,192,28,28],f32>, !torch.list<int>, !torch.float -> !torch.vtensor<[1,192,30,30],f32>
    %none_544 = torch.constant.none
    %int1_545 = torch.constant.int 1
    %int1_546 = torch.constant.int 1
    %389 = torch.prim.ListConstruct %int1_545, %int1_546 : (!torch.int, !torch.int) -> !torch.list<int>
    %int0_547 = torch.constant.int 0
    %int0_548 = torch.constant.int 0
    %390 = torch.prim.ListConstruct %int0_547, %int0_548 : (!torch.int, !torch.int) -> !torch.list<int>
    %int1_549 = torch.constant.int 1
    %int1_550 = torch.constant.int 1
    %391 = torch.prim.ListConstruct %int1_549, %int1_550 : (!torch.int, !torch.int) -> !torch.list<int>
    %false_551 = torch.constant.bool false
    %int0_552 = torch.constant.int 0
    %int0_553 = torch.constant.int 0
    %392 = torch.prim.ListConstruct %int0_552, %int0_553 : (!torch.int, !torch.int) -> !torch.list<int>
    %int192 = torch.constant.int 192
    %393 = torch.aten.convolution %388, %arg66, %none_544, %389, %390, %391, %false_551, %392, %int192 : !torch.vtensor<[1,192,30,30],f32>, !torch.vtensor<[192,1,3,3],f32>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[1,192,28,28],f32>
    %int6_554 = torch.constant.int 6
    %394 = torch.prims.convert_element_type %arg67, %int6_554 : !torch.vtensor<[192],f32>, !torch.int -> !torch.vtensor<[192],f32>
    %int6_555 = torch.constant.int 6
    %395 = torch.prims.convert_element_type %arg68, %int6_555 : !torch.vtensor<[192],f32>, !torch.int -> !torch.vtensor<[192],f32>
    %float1.000000e-03_556 = torch.constant.float 1.000000e-03
    %int1_557 = torch.constant.int 1
    %396 = torch.aten.add.Scalar %395, %float1.000000e-03_556, %int1_557 : !torch.vtensor<[192],f32>, !torch.float, !torch.int -> !torch.vtensor<[192],f32>
    %397 = torch.aten.sqrt %396 : !torch.vtensor<[192],f32> -> !torch.vtensor<[192],f32>
    %398 = torch.aten.reciprocal %397 : !torch.vtensor<[192],f32> -> !torch.vtensor<[192],f32>
    %int1_558 = torch.constant.int 1
    %399 = torch.aten.mul.Scalar %398, %int1_558 : !torch.vtensor<[192],f32>, !torch.int -> !torch.vtensor<[192],f32>
    %int0_559 = torch.constant.int 0
    %400 = torch.prim.ListConstruct %int0_559 : (!torch.int) -> !torch.list<int>
    %none_560 = torch.constant.none
    %none_561 = torch.constant.none
    %none_562 = torch.constant.none
    %false_563 = torch.constant.bool false
    %401 = torch.aten.new_zeros %393, %400, %none_560, %none_561, %none_562, %false_563 : !torch.vtensor<[1,192,28,28],f32>, !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[0],f32>
    %int0_564 = torch.constant.int 0
    %402 = torch.prim.ListConstruct %int0_564 : (!torch.int) -> !torch.list<int>
    %none_565 = torch.constant.none
    %none_566 = torch.constant.none
    %none_567 = torch.constant.none
    %false_568 = torch.constant.bool false
    %403 = torch.aten.new_zeros %393, %402, %none_565, %none_566, %none_567, %false_568 : !torch.vtensor<[1,192,28,28],f32>, !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[0],f32>
    %int-1_569 = torch.constant.int -1
    %404 = torch.aten.unsqueeze %394, %int-1_569 : !torch.vtensor<[192],f32>, !torch.int -> !torch.vtensor<[192,1],f32>
    %int-1_570 = torch.constant.int -1
    %405 = torch.aten.unsqueeze %404, %int-1_570 : !torch.vtensor<[192,1],f32>, !torch.int -> !torch.vtensor<[192,1,1],f32>
    %int-1_571 = torch.constant.int -1
    %406 = torch.aten.unsqueeze %399, %int-1_571 : !torch.vtensor<[192],f32>, !torch.int -> !torch.vtensor<[192,1],f32>
    %int-1_572 = torch.constant.int -1
    %407 = torch.aten.unsqueeze %406, %int-1_572 : !torch.vtensor<[192,1],f32>, !torch.int -> !torch.vtensor<[192,1,1],f32>
    %int1_573 = torch.constant.int 1
    %408 = torch.aten.sub.Tensor %393, %405, %int1_573 : !torch.vtensor<[1,192,28,28],f32>, !torch.vtensor<[192,1,1],f32>, !torch.int -> !torch.vtensor<[1,192,28,28],f32>
    %409 = torch.aten.mul.Tensor %408, %407 : !torch.vtensor<[1,192,28,28],f32>, !torch.vtensor<[192,1,1],f32> -> !torch.vtensor<[1,192,28,28],f32>
    %int-1_574 = torch.constant.int -1
    %410 = torch.aten.unsqueeze %arg69, %int-1_574 : !torch.vtensor<[192],f32>, !torch.int -> !torch.vtensor<[192,1],f32>
    %int-1_575 = torch.constant.int -1
    %411 = torch.aten.unsqueeze %410, %int-1_575 : !torch.vtensor<[192,1],f32>, !torch.int -> !torch.vtensor<[192,1,1],f32>
    %412 = torch.aten.mul.Tensor %409, %411 : !torch.vtensor<[1,192,28,28],f32>, !torch.vtensor<[192,1,1],f32> -> !torch.vtensor<[1,192,28,28],f32>
    %int-1_576 = torch.constant.int -1
    %413 = torch.aten.unsqueeze %arg70, %int-1_576 : !torch.vtensor<[192],f32>, !torch.int -> !torch.vtensor<[192,1],f32>
    %int-1_577 = torch.constant.int -1
    %414 = torch.aten.unsqueeze %413, %int-1_577 : !torch.vtensor<[192,1],f32>, !torch.int -> !torch.vtensor<[192,1,1],f32>
    %int1_578 = torch.constant.int 1
    %415 = torch.aten.add.Tensor %412, %414, %int1_578 : !torch.vtensor<[1,192,28,28],f32>, !torch.vtensor<[192,1,1],f32>, !torch.int -> !torch.vtensor<[1,192,28,28],f32>
    %float0.000000e00_579 = torch.constant.float 0.000000e+00
    %float6.000000e00_580 = torch.constant.float 6.000000e+00
    %416 = torch.aten.hardtanh %415, %float0.000000e00_579, %float6.000000e00_580 : !torch.vtensor<[1,192,28,28],f32>, !torch.float, !torch.float -> !torch.vtensor<[1,192,28,28],f32>
    %int0_581 = torch.constant.int 0
    %int0_582 = torch.constant.int 0
    %int0_583 = torch.constant.int 0
    %int0_584 = torch.constant.int 0
    %417 = torch.prim.ListConstruct %int0_581, %int0_582, %int0_583, %int0_584 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %float0.000000e00_585 = torch.constant.float 0.000000e+00
    %418 = torch.aten.constant_pad_nd %416, %417, %float0.000000e00_585 : !torch.vtensor<[1,192,28,28],f32>, !torch.list<int>, !torch.float -> !torch.vtensor<[1,192,28,28],f32>
    %none_586 = torch.constant.none
    %int1_587 = torch.constant.int 1
    %int1_588 = torch.constant.int 1
    %419 = torch.prim.ListConstruct %int1_587, %int1_588 : (!torch.int, !torch.int) -> !torch.list<int>
    %int0_589 = torch.constant.int 0
    %int0_590 = torch.constant.int 0
    %420 = torch.prim.ListConstruct %int0_589, %int0_590 : (!torch.int, !torch.int) -> !torch.list<int>
    %int1_591 = torch.constant.int 1
    %int1_592 = torch.constant.int 1
    %421 = torch.prim.ListConstruct %int1_591, %int1_592 : (!torch.int, !torch.int) -> !torch.list<int>
    %false_593 = torch.constant.bool false
    %int0_594 = torch.constant.int 0
    %int0_595 = torch.constant.int 0
    %422 = torch.prim.ListConstruct %int0_594, %int0_595 : (!torch.int, !torch.int) -> !torch.list<int>
    %int1_596 = torch.constant.int 1
    %423 = torch.aten.convolution %418, %arg71, %none_586, %419, %420, %421, %false_593, %422, %int1_596 : !torch.vtensor<[1,192,28,28],f32>, !torch.vtensor<[32,192,1,1],f32>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[1,32,28,28],f32>
    %int6_597 = torch.constant.int 6
    %424 = torch.prims.convert_element_type %arg72, %int6_597 : !torch.vtensor<[32],f32>, !torch.int -> !torch.vtensor<[32],f32>
    %int6_598 = torch.constant.int 6
    %425 = torch.prims.convert_element_type %arg73, %int6_598 : !torch.vtensor<[32],f32>, !torch.int -> !torch.vtensor<[32],f32>
    %float1.000000e-03_599 = torch.constant.float 1.000000e-03
    %int1_600 = torch.constant.int 1
    %426 = torch.aten.add.Scalar %425, %float1.000000e-03_599, %int1_600 : !torch.vtensor<[32],f32>, !torch.float, !torch.int -> !torch.vtensor<[32],f32>
    %427 = torch.aten.sqrt %426 : !torch.vtensor<[32],f32> -> !torch.vtensor<[32],f32>
    %428 = torch.aten.reciprocal %427 : !torch.vtensor<[32],f32> -> !torch.vtensor<[32],f32>
    %int1_601 = torch.constant.int 1
    %429 = torch.aten.mul.Scalar %428, %int1_601 : !torch.vtensor<[32],f32>, !torch.int -> !torch.vtensor<[32],f32>
    %int0_602 = torch.constant.int 0
    %430 = torch.prim.ListConstruct %int0_602 : (!torch.int) -> !torch.list<int>
    %none_603 = torch.constant.none
    %none_604 = torch.constant.none
    %none_605 = torch.constant.none
    %false_606 = torch.constant.bool false
    %431 = torch.aten.new_zeros %423, %430, %none_603, %none_604, %none_605, %false_606 : !torch.vtensor<[1,32,28,28],f32>, !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[0],f32>
    %int0_607 = torch.constant.int 0
    %432 = torch.prim.ListConstruct %int0_607 : (!torch.int) -> !torch.list<int>
    %none_608 = torch.constant.none
    %none_609 = torch.constant.none
    %none_610 = torch.constant.none
    %false_611 = torch.constant.bool false
    %433 = torch.aten.new_zeros %423, %432, %none_608, %none_609, %none_610, %false_611 : !torch.vtensor<[1,32,28,28],f32>, !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[0],f32>
    %int-1_612 = torch.constant.int -1
    %434 = torch.aten.unsqueeze %424, %int-1_612 : !torch.vtensor<[32],f32>, !torch.int -> !torch.vtensor<[32,1],f32>
    %int-1_613 = torch.constant.int -1
    %435 = torch.aten.unsqueeze %434, %int-1_613 : !torch.vtensor<[32,1],f32>, !torch.int -> !torch.vtensor<[32,1,1],f32>
    %int-1_614 = torch.constant.int -1
    %436 = torch.aten.unsqueeze %429, %int-1_614 : !torch.vtensor<[32],f32>, !torch.int -> !torch.vtensor<[32,1],f32>
    %int-1_615 = torch.constant.int -1
    %437 = torch.aten.unsqueeze %436, %int-1_615 : !torch.vtensor<[32,1],f32>, !torch.int -> !torch.vtensor<[32,1,1],f32>
    %int1_616 = torch.constant.int 1
    %438 = torch.aten.sub.Tensor %423, %435, %int1_616 : !torch.vtensor<[1,32,28,28],f32>, !torch.vtensor<[32,1,1],f32>, !torch.int -> !torch.vtensor<[1,32,28,28],f32>
    %439 = torch.aten.mul.Tensor %438, %437 : !torch.vtensor<[1,32,28,28],f32>, !torch.vtensor<[32,1,1],f32> -> !torch.vtensor<[1,32,28,28],f32>
    %int-1_617 = torch.constant.int -1
    %440 = torch.aten.unsqueeze %arg74, %int-1_617 : !torch.vtensor<[32],f32>, !torch.int -> !torch.vtensor<[32,1],f32>
    %int-1_618 = torch.constant.int -1
    %441 = torch.aten.unsqueeze %440, %int-1_618 : !torch.vtensor<[32,1],f32>, !torch.int -> !torch.vtensor<[32,1,1],f32>
    %442 = torch.aten.mul.Tensor %439, %441 : !torch.vtensor<[1,32,28,28],f32>, !torch.vtensor<[32,1,1],f32> -> !torch.vtensor<[1,32,28,28],f32>
    %int-1_619 = torch.constant.int -1
    %443 = torch.aten.unsqueeze %arg75, %int-1_619 : !torch.vtensor<[32],f32>, !torch.int -> !torch.vtensor<[32,1],f32>
    %int-1_620 = torch.constant.int -1
    %444 = torch.aten.unsqueeze %443, %int-1_620 : !torch.vtensor<[32,1],f32>, !torch.int -> !torch.vtensor<[32,1,1],f32>
    %int1_621 = torch.constant.int 1
    %445 = torch.aten.add.Tensor %442, %444, %int1_621 : !torch.vtensor<[1,32,28,28],f32>, !torch.vtensor<[32,1,1],f32>, !torch.int -> !torch.vtensor<[1,32,28,28],f32>
    %int1_622 = torch.constant.int 1
    %446 = torch.aten.add.Tensor %356, %445, %int1_622 : !torch.vtensor<[1,32,28,28],f32>, !torch.vtensor<[1,32,28,28],f32>, !torch.int -> !torch.vtensor<[1,32,28,28],f32>
    %int0_623 = torch.constant.int 0
    %int0_624 = torch.constant.int 0
    %int0_625 = torch.constant.int 0
    %int0_626 = torch.constant.int 0
    %447 = torch.prim.ListConstruct %int0_623, %int0_624, %int0_625, %int0_626 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %float0.000000e00_627 = torch.constant.float 0.000000e+00
    %448 = torch.aten.constant_pad_nd %446, %447, %float0.000000e00_627 : !torch.vtensor<[1,32,28,28],f32>, !torch.list<int>, !torch.float -> !torch.vtensor<[1,32,28,28],f32>
    %none_628 = torch.constant.none
    %int1_629 = torch.constant.int 1
    %int1_630 = torch.constant.int 1
    %449 = torch.prim.ListConstruct %int1_629, %int1_630 : (!torch.int, !torch.int) -> !torch.list<int>
    %int0_631 = torch.constant.int 0
    %int0_632 = torch.constant.int 0
    %450 = torch.prim.ListConstruct %int0_631, %int0_632 : (!torch.int, !torch.int) -> !torch.list<int>
    %int1_633 = torch.constant.int 1
    %int1_634 = torch.constant.int 1
    %451 = torch.prim.ListConstruct %int1_633, %int1_634 : (!torch.int, !torch.int) -> !torch.list<int>
    %false_635 = torch.constant.bool false
    %int0_636 = torch.constant.int 0
    %int0_637 = torch.constant.int 0
    %452 = torch.prim.ListConstruct %int0_636, %int0_637 : (!torch.int, !torch.int) -> !torch.list<int>
    %int1_638 = torch.constant.int 1
    %453 = torch.aten.convolution %448, %arg76, %none_628, %449, %450, %451, %false_635, %452, %int1_638 : !torch.vtensor<[1,32,28,28],f32>, !torch.vtensor<[192,32,1,1],f32>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[1,192,28,28],f32>
    %int6_639 = torch.constant.int 6
    %454 = torch.prims.convert_element_type %arg77, %int6_639 : !torch.vtensor<[192],f32>, !torch.int -> !torch.vtensor<[192],f32>
    %int6_640 = torch.constant.int 6
    %455 = torch.prims.convert_element_type %arg78, %int6_640 : !torch.vtensor<[192],f32>, !torch.int -> !torch.vtensor<[192],f32>
    %float1.000000e-03_641 = torch.constant.float 1.000000e-03
    %int1_642 = torch.constant.int 1
    %456 = torch.aten.add.Scalar %455, %float1.000000e-03_641, %int1_642 : !torch.vtensor<[192],f32>, !torch.float, !torch.int -> !torch.vtensor<[192],f32>
    %457 = torch.aten.sqrt %456 : !torch.vtensor<[192],f32> -> !torch.vtensor<[192],f32>
    %458 = torch.aten.reciprocal %457 : !torch.vtensor<[192],f32> -> !torch.vtensor<[192],f32>
    %int1_643 = torch.constant.int 1
    %459 = torch.aten.mul.Scalar %458, %int1_643 : !torch.vtensor<[192],f32>, !torch.int -> !torch.vtensor<[192],f32>
    %int0_644 = torch.constant.int 0
    %460 = torch.prim.ListConstruct %int0_644 : (!torch.int) -> !torch.list<int>
    %none_645 = torch.constant.none
    %none_646 = torch.constant.none
    %none_647 = torch.constant.none
    %false_648 = torch.constant.bool false
    %461 = torch.aten.new_zeros %453, %460, %none_645, %none_646, %none_647, %false_648 : !torch.vtensor<[1,192,28,28],f32>, !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[0],f32>
    %int0_649 = torch.constant.int 0
    %462 = torch.prim.ListConstruct %int0_649 : (!torch.int) -> !torch.list<int>
    %none_650 = torch.constant.none
    %none_651 = torch.constant.none
    %none_652 = torch.constant.none
    %false_653 = torch.constant.bool false
    %463 = torch.aten.new_zeros %453, %462, %none_650, %none_651, %none_652, %false_653 : !torch.vtensor<[1,192,28,28],f32>, !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[0],f32>
    %int-1_654 = torch.constant.int -1
    %464 = torch.aten.unsqueeze %454, %int-1_654 : !torch.vtensor<[192],f32>, !torch.int -> !torch.vtensor<[192,1],f32>
    %int-1_655 = torch.constant.int -1
    %465 = torch.aten.unsqueeze %464, %int-1_655 : !torch.vtensor<[192,1],f32>, !torch.int -> !torch.vtensor<[192,1,1],f32>
    %int-1_656 = torch.constant.int -1
    %466 = torch.aten.unsqueeze %459, %int-1_656 : !torch.vtensor<[192],f32>, !torch.int -> !torch.vtensor<[192,1],f32>
    %int-1_657 = torch.constant.int -1
    %467 = torch.aten.unsqueeze %466, %int-1_657 : !torch.vtensor<[192,1],f32>, !torch.int -> !torch.vtensor<[192,1,1],f32>
    %int1_658 = torch.constant.int 1
    %468 = torch.aten.sub.Tensor %453, %465, %int1_658 : !torch.vtensor<[1,192,28,28],f32>, !torch.vtensor<[192,1,1],f32>, !torch.int -> !torch.vtensor<[1,192,28,28],f32>
    %469 = torch.aten.mul.Tensor %468, %467 : !torch.vtensor<[1,192,28,28],f32>, !torch.vtensor<[192,1,1],f32> -> !torch.vtensor<[1,192,28,28],f32>
    %int-1_659 = torch.constant.int -1
    %470 = torch.aten.unsqueeze %arg79, %int-1_659 : !torch.vtensor<[192],f32>, !torch.int -> !torch.vtensor<[192,1],f32>
    %int-1_660 = torch.constant.int -1
    %471 = torch.aten.unsqueeze %470, %int-1_660 : !torch.vtensor<[192,1],f32>, !torch.int -> !torch.vtensor<[192,1,1],f32>
    %472 = torch.aten.mul.Tensor %469, %471 : !torch.vtensor<[1,192,28,28],f32>, !torch.vtensor<[192,1,1],f32> -> !torch.vtensor<[1,192,28,28],f32>
    %int-1_661 = torch.constant.int -1
    %473 = torch.aten.unsqueeze %arg80, %int-1_661 : !torch.vtensor<[192],f32>, !torch.int -> !torch.vtensor<[192,1],f32>
    %int-1_662 = torch.constant.int -1
    %474 = torch.aten.unsqueeze %473, %int-1_662 : !torch.vtensor<[192,1],f32>, !torch.int -> !torch.vtensor<[192,1,1],f32>
    %int1_663 = torch.constant.int 1
    %475 = torch.aten.add.Tensor %472, %474, %int1_663 : !torch.vtensor<[1,192,28,28],f32>, !torch.vtensor<[192,1,1],f32>, !torch.int -> !torch.vtensor<[1,192,28,28],f32>
    %float0.000000e00_664 = torch.constant.float 0.000000e+00
    %float6.000000e00_665 = torch.constant.float 6.000000e+00
    %476 = torch.aten.hardtanh %475, %float0.000000e00_664, %float6.000000e00_665 : !torch.vtensor<[1,192,28,28],f32>, !torch.float, !torch.float -> !torch.vtensor<[1,192,28,28],f32>
    %int1_666 = torch.constant.int 1
    %int1_667 = torch.constant.int 1
    %int1_668 = torch.constant.int 1
    %int1_669 = torch.constant.int 1
    %477 = torch.prim.ListConstruct %int1_666, %int1_667, %int1_668, %int1_669 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %float0.000000e00_670 = torch.constant.float 0.000000e+00
    %478 = torch.aten.constant_pad_nd %476, %477, %float0.000000e00_670 : !torch.vtensor<[1,192,28,28],f32>, !torch.list<int>, !torch.float -> !torch.vtensor<[1,192,30,30],f32>
    %none_671 = torch.constant.none
    %int1_672 = torch.constant.int 1
    %int1_673 = torch.constant.int 1
    %479 = torch.prim.ListConstruct %int1_672, %int1_673 : (!torch.int, !torch.int) -> !torch.list<int>
    %int0_674 = torch.constant.int 0
    %int0_675 = torch.constant.int 0
    %480 = torch.prim.ListConstruct %int0_674, %int0_675 : (!torch.int, !torch.int) -> !torch.list<int>
    %int1_676 = torch.constant.int 1
    %int1_677 = torch.constant.int 1
    %481 = torch.prim.ListConstruct %int1_676, %int1_677 : (!torch.int, !torch.int) -> !torch.list<int>
    %false_678 = torch.constant.bool false
    %int0_679 = torch.constant.int 0
    %int0_680 = torch.constant.int 0
    %482 = torch.prim.ListConstruct %int0_679, %int0_680 : (!torch.int, !torch.int) -> !torch.list<int>
    %int192_681 = torch.constant.int 192
    %483 = torch.aten.convolution %478, %arg81, %none_671, %479, %480, %481, %false_678, %482, %int192_681 : !torch.vtensor<[1,192,30,30],f32>, !torch.vtensor<[192,1,3,3],f32>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[1,192,28,28],f32>
    %int6_682 = torch.constant.int 6
    %484 = torch.prims.convert_element_type %arg82, %int6_682 : !torch.vtensor<[192],f32>, !torch.int -> !torch.vtensor<[192],f32>
    %int6_683 = torch.constant.int 6
    %485 = torch.prims.convert_element_type %arg83, %int6_683 : !torch.vtensor<[192],f32>, !torch.int -> !torch.vtensor<[192],f32>
    %float1.000000e-03_684 = torch.constant.float 1.000000e-03
    %int1_685 = torch.constant.int 1
    %486 = torch.aten.add.Scalar %485, %float1.000000e-03_684, %int1_685 : !torch.vtensor<[192],f32>, !torch.float, !torch.int -> !torch.vtensor<[192],f32>
    %487 = torch.aten.sqrt %486 : !torch.vtensor<[192],f32> -> !torch.vtensor<[192],f32>
    %488 = torch.aten.reciprocal %487 : !torch.vtensor<[192],f32> -> !torch.vtensor<[192],f32>
    %int1_686 = torch.constant.int 1
    %489 = torch.aten.mul.Scalar %488, %int1_686 : !torch.vtensor<[192],f32>, !torch.int -> !torch.vtensor<[192],f32>
    %int0_687 = torch.constant.int 0
    %490 = torch.prim.ListConstruct %int0_687 : (!torch.int) -> !torch.list<int>
    %none_688 = torch.constant.none
    %none_689 = torch.constant.none
    %none_690 = torch.constant.none
    %false_691 = torch.constant.bool false
    %491 = torch.aten.new_zeros %483, %490, %none_688, %none_689, %none_690, %false_691 : !torch.vtensor<[1,192,28,28],f32>, !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[0],f32>
    %int0_692 = torch.constant.int 0
    %492 = torch.prim.ListConstruct %int0_692 : (!torch.int) -> !torch.list<int>
    %none_693 = torch.constant.none
    %none_694 = torch.constant.none
    %none_695 = torch.constant.none
    %false_696 = torch.constant.bool false
    %493 = torch.aten.new_zeros %483, %492, %none_693, %none_694, %none_695, %false_696 : !torch.vtensor<[1,192,28,28],f32>, !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[0],f32>
    %int-1_697 = torch.constant.int -1
    %494 = torch.aten.unsqueeze %484, %int-1_697 : !torch.vtensor<[192],f32>, !torch.int -> !torch.vtensor<[192,1],f32>
    %int-1_698 = torch.constant.int -1
    %495 = torch.aten.unsqueeze %494, %int-1_698 : !torch.vtensor<[192,1],f32>, !torch.int -> !torch.vtensor<[192,1,1],f32>
    %int-1_699 = torch.constant.int -1
    %496 = torch.aten.unsqueeze %489, %int-1_699 : !torch.vtensor<[192],f32>, !torch.int -> !torch.vtensor<[192,1],f32>
    %int-1_700 = torch.constant.int -1
    %497 = torch.aten.unsqueeze %496, %int-1_700 : !torch.vtensor<[192,1],f32>, !torch.int -> !torch.vtensor<[192,1,1],f32>
    %int1_701 = torch.constant.int 1
    %498 = torch.aten.sub.Tensor %483, %495, %int1_701 : !torch.vtensor<[1,192,28,28],f32>, !torch.vtensor<[192,1,1],f32>, !torch.int -> !torch.vtensor<[1,192,28,28],f32>
    %499 = torch.aten.mul.Tensor %498, %497 : !torch.vtensor<[1,192,28,28],f32>, !torch.vtensor<[192,1,1],f32> -> !torch.vtensor<[1,192,28,28],f32>
    %int-1_702 = torch.constant.int -1
    %500 = torch.aten.unsqueeze %arg84, %int-1_702 : !torch.vtensor<[192],f32>, !torch.int -> !torch.vtensor<[192,1],f32>
    %int-1_703 = torch.constant.int -1
    %501 = torch.aten.unsqueeze %500, %int-1_703 : !torch.vtensor<[192,1],f32>, !torch.int -> !torch.vtensor<[192,1,1],f32>
    %502 = torch.aten.mul.Tensor %499, %501 : !torch.vtensor<[1,192,28,28],f32>, !torch.vtensor<[192,1,1],f32> -> !torch.vtensor<[1,192,28,28],f32>
    %int-1_704 = torch.constant.int -1
    %503 = torch.aten.unsqueeze %arg85, %int-1_704 : !torch.vtensor<[192],f32>, !torch.int -> !torch.vtensor<[192,1],f32>
    %int-1_705 = torch.constant.int -1
    %504 = torch.aten.unsqueeze %503, %int-1_705 : !torch.vtensor<[192,1],f32>, !torch.int -> !torch.vtensor<[192,1,1],f32>
    %int1_706 = torch.constant.int 1
    %505 = torch.aten.add.Tensor %502, %504, %int1_706 : !torch.vtensor<[1,192,28,28],f32>, !torch.vtensor<[192,1,1],f32>, !torch.int -> !torch.vtensor<[1,192,28,28],f32>
    %float0.000000e00_707 = torch.constant.float 0.000000e+00
    %float6.000000e00_708 = torch.constant.float 6.000000e+00
    %506 = torch.aten.hardtanh %505, %float0.000000e00_707, %float6.000000e00_708 : !torch.vtensor<[1,192,28,28],f32>, !torch.float, !torch.float -> !torch.vtensor<[1,192,28,28],f32>
    %int0_709 = torch.constant.int 0
    %int0_710 = torch.constant.int 0
    %int0_711 = torch.constant.int 0
    %int0_712 = torch.constant.int 0
    %507 = torch.prim.ListConstruct %int0_709, %int0_710, %int0_711, %int0_712 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %float0.000000e00_713 = torch.constant.float 0.000000e+00
    %508 = torch.aten.constant_pad_nd %506, %507, %float0.000000e00_713 : !torch.vtensor<[1,192,28,28],f32>, !torch.list<int>, !torch.float -> !torch.vtensor<[1,192,28,28],f32>
    %none_714 = torch.constant.none
    %int1_715 = torch.constant.int 1
    %int1_716 = torch.constant.int 1
    %509 = torch.prim.ListConstruct %int1_715, %int1_716 : (!torch.int, !torch.int) -> !torch.list<int>
    %int0_717 = torch.constant.int 0
    %int0_718 = torch.constant.int 0
    %510 = torch.prim.ListConstruct %int0_717, %int0_718 : (!torch.int, !torch.int) -> !torch.list<int>
    %int1_719 = torch.constant.int 1
    %int1_720 = torch.constant.int 1
    %511 = torch.prim.ListConstruct %int1_719, %int1_720 : (!torch.int, !torch.int) -> !torch.list<int>
    %false_721 = torch.constant.bool false
    %int0_722 = torch.constant.int 0
    %int0_723 = torch.constant.int 0
    %512 = torch.prim.ListConstruct %int0_722, %int0_723 : (!torch.int, !torch.int) -> !torch.list<int>
    %int1_724 = torch.constant.int 1
    %513 = torch.aten.convolution %508, %arg86, %none_714, %509, %510, %511, %false_721, %512, %int1_724 : !torch.vtensor<[1,192,28,28],f32>, !torch.vtensor<[32,192,1,1],f32>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[1,32,28,28],f32>
    %int6_725 = torch.constant.int 6
    %514 = torch.prims.convert_element_type %arg87, %int6_725 : !torch.vtensor<[32],f32>, !torch.int -> !torch.vtensor<[32],f32>
    %int6_726 = torch.constant.int 6
    %515 = torch.prims.convert_element_type %arg88, %int6_726 : !torch.vtensor<[32],f32>, !torch.int -> !torch.vtensor<[32],f32>
    %float1.000000e-03_727 = torch.constant.float 1.000000e-03
    %int1_728 = torch.constant.int 1
    %516 = torch.aten.add.Scalar %515, %float1.000000e-03_727, %int1_728 : !torch.vtensor<[32],f32>, !torch.float, !torch.int -> !torch.vtensor<[32],f32>
    %517 = torch.aten.sqrt %516 : !torch.vtensor<[32],f32> -> !torch.vtensor<[32],f32>
    %518 = torch.aten.reciprocal %517 : !torch.vtensor<[32],f32> -> !torch.vtensor<[32],f32>
    %int1_729 = torch.constant.int 1
    %519 = torch.aten.mul.Scalar %518, %int1_729 : !torch.vtensor<[32],f32>, !torch.int -> !torch.vtensor<[32],f32>
    %int0_730 = torch.constant.int 0
    %520 = torch.prim.ListConstruct %int0_730 : (!torch.int) -> !torch.list<int>
    %none_731 = torch.constant.none
    %none_732 = torch.constant.none
    %none_733 = torch.constant.none
    %false_734 = torch.constant.bool false
    %521 = torch.aten.new_zeros %513, %520, %none_731, %none_732, %none_733, %false_734 : !torch.vtensor<[1,32,28,28],f32>, !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[0],f32>
    %int0_735 = torch.constant.int 0
    %522 = torch.prim.ListConstruct %int0_735 : (!torch.int) -> !torch.list<int>
    %none_736 = torch.constant.none
    %none_737 = torch.constant.none
    %none_738 = torch.constant.none
    %false_739 = torch.constant.bool false
    %523 = torch.aten.new_zeros %513, %522, %none_736, %none_737, %none_738, %false_739 : !torch.vtensor<[1,32,28,28],f32>, !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[0],f32>
    %int-1_740 = torch.constant.int -1
    %524 = torch.aten.unsqueeze %514, %int-1_740 : !torch.vtensor<[32],f32>, !torch.int -> !torch.vtensor<[32,1],f32>
    %int-1_741 = torch.constant.int -1
    %525 = torch.aten.unsqueeze %524, %int-1_741 : !torch.vtensor<[32,1],f32>, !torch.int -> !torch.vtensor<[32,1,1],f32>
    %int-1_742 = torch.constant.int -1
    %526 = torch.aten.unsqueeze %519, %int-1_742 : !torch.vtensor<[32],f32>, !torch.int -> !torch.vtensor<[32,1],f32>
    %int-1_743 = torch.constant.int -1
    %527 = torch.aten.unsqueeze %526, %int-1_743 : !torch.vtensor<[32,1],f32>, !torch.int -> !torch.vtensor<[32,1,1],f32>
    %int1_744 = torch.constant.int 1
    %528 = torch.aten.sub.Tensor %513, %525, %int1_744 : !torch.vtensor<[1,32,28,28],f32>, !torch.vtensor<[32,1,1],f32>, !torch.int -> !torch.vtensor<[1,32,28,28],f32>
    %529 = torch.aten.mul.Tensor %528, %527 : !torch.vtensor<[1,32,28,28],f32>, !torch.vtensor<[32,1,1],f32> -> !torch.vtensor<[1,32,28,28],f32>
    %int-1_745 = torch.constant.int -1
    %530 = torch.aten.unsqueeze %arg89, %int-1_745 : !torch.vtensor<[32],f32>, !torch.int -> !torch.vtensor<[32,1],f32>
    %int-1_746 = torch.constant.int -1
    %531 = torch.aten.unsqueeze %530, %int-1_746 : !torch.vtensor<[32,1],f32>, !torch.int -> !torch.vtensor<[32,1,1],f32>
    %532 = torch.aten.mul.Tensor %529, %531 : !torch.vtensor<[1,32,28,28],f32>, !torch.vtensor<[32,1,1],f32> -> !torch.vtensor<[1,32,28,28],f32>
    %int-1_747 = torch.constant.int -1
    %533 = torch.aten.unsqueeze %arg90, %int-1_747 : !torch.vtensor<[32],f32>, !torch.int -> !torch.vtensor<[32,1],f32>
    %int-1_748 = torch.constant.int -1
    %534 = torch.aten.unsqueeze %533, %int-1_748 : !torch.vtensor<[32,1],f32>, !torch.int -> !torch.vtensor<[32,1,1],f32>
    %int1_749 = torch.constant.int 1
    %535 = torch.aten.add.Tensor %532, %534, %int1_749 : !torch.vtensor<[1,32,28,28],f32>, !torch.vtensor<[32,1,1],f32>, !torch.int -> !torch.vtensor<[1,32,28,28],f32>
    %int1_750 = torch.constant.int 1
    %536 = torch.aten.add.Tensor %446, %535, %int1_750 : !torch.vtensor<[1,32,28,28],f32>, !torch.vtensor<[1,32,28,28],f32>, !torch.int -> !torch.vtensor<[1,32,28,28],f32>
    %int0_751 = torch.constant.int 0
    %int0_752 = torch.constant.int 0
    %int0_753 = torch.constant.int 0
    %int0_754 = torch.constant.int 0
    %537 = torch.prim.ListConstruct %int0_751, %int0_752, %int0_753, %int0_754 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %float0.000000e00_755 = torch.constant.float 0.000000e+00
    %538 = torch.aten.constant_pad_nd %536, %537, %float0.000000e00_755 : !torch.vtensor<[1,32,28,28],f32>, !torch.list<int>, !torch.float -> !torch.vtensor<[1,32,28,28],f32>
    %none_756 = torch.constant.none
    %int1_757 = torch.constant.int 1
    %int1_758 = torch.constant.int 1
    %539 = torch.prim.ListConstruct %int1_757, %int1_758 : (!torch.int, !torch.int) -> !torch.list<int>
    %int0_759 = torch.constant.int 0
    %int0_760 = torch.constant.int 0
    %540 = torch.prim.ListConstruct %int0_759, %int0_760 : (!torch.int, !torch.int) -> !torch.list<int>
    %int1_761 = torch.constant.int 1
    %int1_762 = torch.constant.int 1
    %541 = torch.prim.ListConstruct %int1_761, %int1_762 : (!torch.int, !torch.int) -> !torch.list<int>
    %false_763 = torch.constant.bool false
    %int0_764 = torch.constant.int 0
    %int0_765 = torch.constant.int 0
    %542 = torch.prim.ListConstruct %int0_764, %int0_765 : (!torch.int, !torch.int) -> !torch.list<int>
    %int1_766 = torch.constant.int 1
    %543 = torch.aten.convolution %538, %arg91, %none_756, %539, %540, %541, %false_763, %542, %int1_766 : !torch.vtensor<[1,32,28,28],f32>, !torch.vtensor<[192,32,1,1],f32>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[1,192,28,28],f32>
    %int6_767 = torch.constant.int 6
    %544 = torch.prims.convert_element_type %arg92, %int6_767 : !torch.vtensor<[192],f32>, !torch.int -> !torch.vtensor<[192],f32>
    %int6_768 = torch.constant.int 6
    %545 = torch.prims.convert_element_type %arg93, %int6_768 : !torch.vtensor<[192],f32>, !torch.int -> !torch.vtensor<[192],f32>
    %float1.000000e-03_769 = torch.constant.float 1.000000e-03
    %int1_770 = torch.constant.int 1
    %546 = torch.aten.add.Scalar %545, %float1.000000e-03_769, %int1_770 : !torch.vtensor<[192],f32>, !torch.float, !torch.int -> !torch.vtensor<[192],f32>
    %547 = torch.aten.sqrt %546 : !torch.vtensor<[192],f32> -> !torch.vtensor<[192],f32>
    %548 = torch.aten.reciprocal %547 : !torch.vtensor<[192],f32> -> !torch.vtensor<[192],f32>
    %int1_771 = torch.constant.int 1
    %549 = torch.aten.mul.Scalar %548, %int1_771 : !torch.vtensor<[192],f32>, !torch.int -> !torch.vtensor<[192],f32>
    %int0_772 = torch.constant.int 0
    %550 = torch.prim.ListConstruct %int0_772 : (!torch.int) -> !torch.list<int>
    %none_773 = torch.constant.none
    %none_774 = torch.constant.none
    %none_775 = torch.constant.none
    %false_776 = torch.constant.bool false
    %551 = torch.aten.new_zeros %543, %550, %none_773, %none_774, %none_775, %false_776 : !torch.vtensor<[1,192,28,28],f32>, !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[0],f32>
    %int0_777 = torch.constant.int 0
    %552 = torch.prim.ListConstruct %int0_777 : (!torch.int) -> !torch.list<int>
    %none_778 = torch.constant.none
    %none_779 = torch.constant.none
    %none_780 = torch.constant.none
    %false_781 = torch.constant.bool false
    %553 = torch.aten.new_zeros %543, %552, %none_778, %none_779, %none_780, %false_781 : !torch.vtensor<[1,192,28,28],f32>, !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[0],f32>
    %int-1_782 = torch.constant.int -1
    %554 = torch.aten.unsqueeze %544, %int-1_782 : !torch.vtensor<[192],f32>, !torch.int -> !torch.vtensor<[192,1],f32>
    %int-1_783 = torch.constant.int -1
    %555 = torch.aten.unsqueeze %554, %int-1_783 : !torch.vtensor<[192,1],f32>, !torch.int -> !torch.vtensor<[192,1,1],f32>
    %int-1_784 = torch.constant.int -1
    %556 = torch.aten.unsqueeze %549, %int-1_784 : !torch.vtensor<[192],f32>, !torch.int -> !torch.vtensor<[192,1],f32>
    %int-1_785 = torch.constant.int -1
    %557 = torch.aten.unsqueeze %556, %int-1_785 : !torch.vtensor<[192,1],f32>, !torch.int -> !torch.vtensor<[192,1,1],f32>
    %int1_786 = torch.constant.int 1
    %558 = torch.aten.sub.Tensor %543, %555, %int1_786 : !torch.vtensor<[1,192,28,28],f32>, !torch.vtensor<[192,1,1],f32>, !torch.int -> !torch.vtensor<[1,192,28,28],f32>
    %559 = torch.aten.mul.Tensor %558, %557 : !torch.vtensor<[1,192,28,28],f32>, !torch.vtensor<[192,1,1],f32> -> !torch.vtensor<[1,192,28,28],f32>
    %int-1_787 = torch.constant.int -1
    %560 = torch.aten.unsqueeze %arg94, %int-1_787 : !torch.vtensor<[192],f32>, !torch.int -> !torch.vtensor<[192,1],f32>
    %int-1_788 = torch.constant.int -1
    %561 = torch.aten.unsqueeze %560, %int-1_788 : !torch.vtensor<[192,1],f32>, !torch.int -> !torch.vtensor<[192,1,1],f32>
    %562 = torch.aten.mul.Tensor %559, %561 : !torch.vtensor<[1,192,28,28],f32>, !torch.vtensor<[192,1,1],f32> -> !torch.vtensor<[1,192,28,28],f32>
    %int-1_789 = torch.constant.int -1
    %563 = torch.aten.unsqueeze %arg95, %int-1_789 : !torch.vtensor<[192],f32>, !torch.int -> !torch.vtensor<[192,1],f32>
    %int-1_790 = torch.constant.int -1
    %564 = torch.aten.unsqueeze %563, %int-1_790 : !torch.vtensor<[192,1],f32>, !torch.int -> !torch.vtensor<[192,1,1],f32>
    %int1_791 = torch.constant.int 1
    %565 = torch.aten.add.Tensor %562, %564, %int1_791 : !torch.vtensor<[1,192,28,28],f32>, !torch.vtensor<[192,1,1],f32>, !torch.int -> !torch.vtensor<[1,192,28,28],f32>
    %float0.000000e00_792 = torch.constant.float 0.000000e+00
    %float6.000000e00_793 = torch.constant.float 6.000000e+00
    %566 = torch.aten.hardtanh %565, %float0.000000e00_792, %float6.000000e00_793 : !torch.vtensor<[1,192,28,28],f32>, !torch.float, !torch.float -> !torch.vtensor<[1,192,28,28],f32>
    %int0_794 = torch.constant.int 0
    %int1_795 = torch.constant.int 1
    %int0_796 = torch.constant.int 0
    %int1_797 = torch.constant.int 1
    %567 = torch.prim.ListConstruct %int0_794, %int1_795, %int0_796, %int1_797 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %float0.000000e00_798 = torch.constant.float 0.000000e+00
    %568 = torch.aten.constant_pad_nd %566, %567, %float0.000000e00_798 : !torch.vtensor<[1,192,28,28],f32>, !torch.list<int>, !torch.float -> !torch.vtensor<[1,192,29,29],f32>
    %none_799 = torch.constant.none
    %int2_800 = torch.constant.int 2
    %int2_801 = torch.constant.int 2
    %569 = torch.prim.ListConstruct %int2_800, %int2_801 : (!torch.int, !torch.int) -> !torch.list<int>
    %int0_802 = torch.constant.int 0
    %int0_803 = torch.constant.int 0
    %570 = torch.prim.ListConstruct %int0_802, %int0_803 : (!torch.int, !torch.int) -> !torch.list<int>
    %int1_804 = torch.constant.int 1
    %int1_805 = torch.constant.int 1
    %571 = torch.prim.ListConstruct %int1_804, %int1_805 : (!torch.int, !torch.int) -> !torch.list<int>
    %false_806 = torch.constant.bool false
    %int0_807 = torch.constant.int 0
    %int0_808 = torch.constant.int 0
    %572 = torch.prim.ListConstruct %int0_807, %int0_808 : (!torch.int, !torch.int) -> !torch.list<int>
    %int192_809 = torch.constant.int 192
    %573 = torch.aten.convolution %568, %arg96, %none_799, %569, %570, %571, %false_806, %572, %int192_809 : !torch.vtensor<[1,192,29,29],f32>, !torch.vtensor<[192,1,3,3],f32>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[1,192,14,14],f32>
    %int6_810 = torch.constant.int 6
    %574 = torch.prims.convert_element_type %arg97, %int6_810 : !torch.vtensor<[192],f32>, !torch.int -> !torch.vtensor<[192],f32>
    %int6_811 = torch.constant.int 6
    %575 = torch.prims.convert_element_type %arg98, %int6_811 : !torch.vtensor<[192],f32>, !torch.int -> !torch.vtensor<[192],f32>
    %float1.000000e-03_812 = torch.constant.float 1.000000e-03
    %int1_813 = torch.constant.int 1
    %576 = torch.aten.add.Scalar %575, %float1.000000e-03_812, %int1_813 : !torch.vtensor<[192],f32>, !torch.float, !torch.int -> !torch.vtensor<[192],f32>
    %577 = torch.aten.sqrt %576 : !torch.vtensor<[192],f32> -> !torch.vtensor<[192],f32>
    %578 = torch.aten.reciprocal %577 : !torch.vtensor<[192],f32> -> !torch.vtensor<[192],f32>
    %int1_814 = torch.constant.int 1
    %579 = torch.aten.mul.Scalar %578, %int1_814 : !torch.vtensor<[192],f32>, !torch.int -> !torch.vtensor<[192],f32>
    %int0_815 = torch.constant.int 0
    %580 = torch.prim.ListConstruct %int0_815 : (!torch.int) -> !torch.list<int>
    %none_816 = torch.constant.none
    %none_817 = torch.constant.none
    %none_818 = torch.constant.none
    %false_819 = torch.constant.bool false
    %581 = torch.aten.new_zeros %573, %580, %none_816, %none_817, %none_818, %false_819 : !torch.vtensor<[1,192,14,14],f32>, !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[0],f32>
    %int0_820 = torch.constant.int 0
    %582 = torch.prim.ListConstruct %int0_820 : (!torch.int) -> !torch.list<int>
    %none_821 = torch.constant.none
    %none_822 = torch.constant.none
    %none_823 = torch.constant.none
    %false_824 = torch.constant.bool false
    %583 = torch.aten.new_zeros %573, %582, %none_821, %none_822, %none_823, %false_824 : !torch.vtensor<[1,192,14,14],f32>, !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[0],f32>
    %int-1_825 = torch.constant.int -1
    %584 = torch.aten.unsqueeze %574, %int-1_825 : !torch.vtensor<[192],f32>, !torch.int -> !torch.vtensor<[192,1],f32>
    %int-1_826 = torch.constant.int -1
    %585 = torch.aten.unsqueeze %584, %int-1_826 : !torch.vtensor<[192,1],f32>, !torch.int -> !torch.vtensor<[192,1,1],f32>
    %int-1_827 = torch.constant.int -1
    %586 = torch.aten.unsqueeze %579, %int-1_827 : !torch.vtensor<[192],f32>, !torch.int -> !torch.vtensor<[192,1],f32>
    %int-1_828 = torch.constant.int -1
    %587 = torch.aten.unsqueeze %586, %int-1_828 : !torch.vtensor<[192,1],f32>, !torch.int -> !torch.vtensor<[192,1,1],f32>
    %int1_829 = torch.constant.int 1
    %588 = torch.aten.sub.Tensor %573, %585, %int1_829 : !torch.vtensor<[1,192,14,14],f32>, !torch.vtensor<[192,1,1],f32>, !torch.int -> !torch.vtensor<[1,192,14,14],f32>
    %589 = torch.aten.mul.Tensor %588, %587 : !torch.vtensor<[1,192,14,14],f32>, !torch.vtensor<[192,1,1],f32> -> !torch.vtensor<[1,192,14,14],f32>
    %int-1_830 = torch.constant.int -1
    %590 = torch.aten.unsqueeze %arg99, %int-1_830 : !torch.vtensor<[192],f32>, !torch.int -> !torch.vtensor<[192,1],f32>
    %int-1_831 = torch.constant.int -1
    %591 = torch.aten.unsqueeze %590, %int-1_831 : !torch.vtensor<[192,1],f32>, !torch.int -> !torch.vtensor<[192,1,1],f32>
    %592 = torch.aten.mul.Tensor %589, %591 : !torch.vtensor<[1,192,14,14],f32>, !torch.vtensor<[192,1,1],f32> -> !torch.vtensor<[1,192,14,14],f32>
    %int-1_832 = torch.constant.int -1
    %593 = torch.aten.unsqueeze %arg100, %int-1_832 : !torch.vtensor<[192],f32>, !torch.int -> !torch.vtensor<[192,1],f32>
    %int-1_833 = torch.constant.int -1
    %594 = torch.aten.unsqueeze %593, %int-1_833 : !torch.vtensor<[192,1],f32>, !torch.int -> !torch.vtensor<[192,1,1],f32>
    %int1_834 = torch.constant.int 1
    %595 = torch.aten.add.Tensor %592, %594, %int1_834 : !torch.vtensor<[1,192,14,14],f32>, !torch.vtensor<[192,1,1],f32>, !torch.int -> !torch.vtensor<[1,192,14,14],f32>
    %float0.000000e00_835 = torch.constant.float 0.000000e+00
    %float6.000000e00_836 = torch.constant.float 6.000000e+00
    %596 = torch.aten.hardtanh %595, %float0.000000e00_835, %float6.000000e00_836 : !torch.vtensor<[1,192,14,14],f32>, !torch.float, !torch.float -> !torch.vtensor<[1,192,14,14],f32>
    %int0_837 = torch.constant.int 0
    %int0_838 = torch.constant.int 0
    %int0_839 = torch.constant.int 0
    %int0_840 = torch.constant.int 0
    %597 = torch.prim.ListConstruct %int0_837, %int0_838, %int0_839, %int0_840 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %float0.000000e00_841 = torch.constant.float 0.000000e+00
    %598 = torch.aten.constant_pad_nd %596, %597, %float0.000000e00_841 : !torch.vtensor<[1,192,14,14],f32>, !torch.list<int>, !torch.float -> !torch.vtensor<[1,192,14,14],f32>
    %none_842 = torch.constant.none
    %int1_843 = torch.constant.int 1
    %int1_844 = torch.constant.int 1
    %599 = torch.prim.ListConstruct %int1_843, %int1_844 : (!torch.int, !torch.int) -> !torch.list<int>
    %int0_845 = torch.constant.int 0
    %int0_846 = torch.constant.int 0
    %600 = torch.prim.ListConstruct %int0_845, %int0_846 : (!torch.int, !torch.int) -> !torch.list<int>
    %int1_847 = torch.constant.int 1
    %int1_848 = torch.constant.int 1
    %601 = torch.prim.ListConstruct %int1_847, %int1_848 : (!torch.int, !torch.int) -> !torch.list<int>
    %false_849 = torch.constant.bool false
    %int0_850 = torch.constant.int 0
    %int0_851 = torch.constant.int 0
    %602 = torch.prim.ListConstruct %int0_850, %int0_851 : (!torch.int, !torch.int) -> !torch.list<int>
    %int1_852 = torch.constant.int 1
    %603 = torch.aten.convolution %598, %arg101, %none_842, %599, %600, %601, %false_849, %602, %int1_852 : !torch.vtensor<[1,192,14,14],f32>, !torch.vtensor<[64,192,1,1],f32>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[1,64,14,14],f32>
    %int6_853 = torch.constant.int 6
    %604 = torch.prims.convert_element_type %arg102, %int6_853 : !torch.vtensor<[64],f32>, !torch.int -> !torch.vtensor<[64],f32>
    %int6_854 = torch.constant.int 6
    %605 = torch.prims.convert_element_type %arg103, %int6_854 : !torch.vtensor<[64],f32>, !torch.int -> !torch.vtensor<[64],f32>
    %float1.000000e-03_855 = torch.constant.float 1.000000e-03
    %int1_856 = torch.constant.int 1
    %606 = torch.aten.add.Scalar %605, %float1.000000e-03_855, %int1_856 : !torch.vtensor<[64],f32>, !torch.float, !torch.int -> !torch.vtensor<[64],f32>
    %607 = torch.aten.sqrt %606 : !torch.vtensor<[64],f32> -> !torch.vtensor<[64],f32>
    %608 = torch.aten.reciprocal %607 : !torch.vtensor<[64],f32> -> !torch.vtensor<[64],f32>
    %int1_857 = torch.constant.int 1
    %609 = torch.aten.mul.Scalar %608, %int1_857 : !torch.vtensor<[64],f32>, !torch.int -> !torch.vtensor<[64],f32>
    %int0_858 = torch.constant.int 0
    %610 = torch.prim.ListConstruct %int0_858 : (!torch.int) -> !torch.list<int>
    %none_859 = torch.constant.none
    %none_860 = torch.constant.none
    %none_861 = torch.constant.none
    %false_862 = torch.constant.bool false
    %611 = torch.aten.new_zeros %603, %610, %none_859, %none_860, %none_861, %false_862 : !torch.vtensor<[1,64,14,14],f32>, !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[0],f32>
    %int0_863 = torch.constant.int 0
    %612 = torch.prim.ListConstruct %int0_863 : (!torch.int) -> !torch.list<int>
    %none_864 = torch.constant.none
    %none_865 = torch.constant.none
    %none_866 = torch.constant.none
    %false_867 = torch.constant.bool false
    %613 = torch.aten.new_zeros %603, %612, %none_864, %none_865, %none_866, %false_867 : !torch.vtensor<[1,64,14,14],f32>, !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[0],f32>
    %int-1_868 = torch.constant.int -1
    %614 = torch.aten.unsqueeze %604, %int-1_868 : !torch.vtensor<[64],f32>, !torch.int -> !torch.vtensor<[64,1],f32>
    %int-1_869 = torch.constant.int -1
    %615 = torch.aten.unsqueeze %614, %int-1_869 : !torch.vtensor<[64,1],f32>, !torch.int -> !torch.vtensor<[64,1,1],f32>
    %int-1_870 = torch.constant.int -1
    %616 = torch.aten.unsqueeze %609, %int-1_870 : !torch.vtensor<[64],f32>, !torch.int -> !torch.vtensor<[64,1],f32>
    %int-1_871 = torch.constant.int -1
    %617 = torch.aten.unsqueeze %616, %int-1_871 : !torch.vtensor<[64,1],f32>, !torch.int -> !torch.vtensor<[64,1,1],f32>
    %int1_872 = torch.constant.int 1
    %618 = torch.aten.sub.Tensor %603, %615, %int1_872 : !torch.vtensor<[1,64,14,14],f32>, !torch.vtensor<[64,1,1],f32>, !torch.int -> !torch.vtensor<[1,64,14,14],f32>
    %619 = torch.aten.mul.Tensor %618, %617 : !torch.vtensor<[1,64,14,14],f32>, !torch.vtensor<[64,1,1],f32> -> !torch.vtensor<[1,64,14,14],f32>
    %int-1_873 = torch.constant.int -1
    %620 = torch.aten.unsqueeze %arg104, %int-1_873 : !torch.vtensor<[64],f32>, !torch.int -> !torch.vtensor<[64,1],f32>
    %int-1_874 = torch.constant.int -1
    %621 = torch.aten.unsqueeze %620, %int-1_874 : !torch.vtensor<[64,1],f32>, !torch.int -> !torch.vtensor<[64,1,1],f32>
    %622 = torch.aten.mul.Tensor %619, %621 : !torch.vtensor<[1,64,14,14],f32>, !torch.vtensor<[64,1,1],f32> -> !torch.vtensor<[1,64,14,14],f32>
    %int-1_875 = torch.constant.int -1
    %623 = torch.aten.unsqueeze %arg105, %int-1_875 : !torch.vtensor<[64],f32>, !torch.int -> !torch.vtensor<[64,1],f32>
    %int-1_876 = torch.constant.int -1
    %624 = torch.aten.unsqueeze %623, %int-1_876 : !torch.vtensor<[64,1],f32>, !torch.int -> !torch.vtensor<[64,1,1],f32>
    %int1_877 = torch.constant.int 1
    %625 = torch.aten.add.Tensor %622, %624, %int1_877 : !torch.vtensor<[1,64,14,14],f32>, !torch.vtensor<[64,1,1],f32>, !torch.int -> !torch.vtensor<[1,64,14,14],f32>
    %int0_878 = torch.constant.int 0
    %int0_879 = torch.constant.int 0
    %int0_880 = torch.constant.int 0
    %int0_881 = torch.constant.int 0
    %626 = torch.prim.ListConstruct %int0_878, %int0_879, %int0_880, %int0_881 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %float0.000000e00_882 = torch.constant.float 0.000000e+00
    %627 = torch.aten.constant_pad_nd %625, %626, %float0.000000e00_882 : !torch.vtensor<[1,64,14,14],f32>, !torch.list<int>, !torch.float -> !torch.vtensor<[1,64,14,14],f32>
    %none_883 = torch.constant.none
    %int1_884 = torch.constant.int 1
    %int1_885 = torch.constant.int 1
    %628 = torch.prim.ListConstruct %int1_884, %int1_885 : (!torch.int, !torch.int) -> !torch.list<int>
    %int0_886 = torch.constant.int 0
    %int0_887 = torch.constant.int 0
    %629 = torch.prim.ListConstruct %int0_886, %int0_887 : (!torch.int, !torch.int) -> !torch.list<int>
    %int1_888 = torch.constant.int 1
    %int1_889 = torch.constant.int 1
    %630 = torch.prim.ListConstruct %int1_888, %int1_889 : (!torch.int, !torch.int) -> !torch.list<int>
    %false_890 = torch.constant.bool false
    %int0_891 = torch.constant.int 0
    %int0_892 = torch.constant.int 0
    %631 = torch.prim.ListConstruct %int0_891, %int0_892 : (!torch.int, !torch.int) -> !torch.list<int>
    %int1_893 = torch.constant.int 1
    %632 = torch.aten.convolution %627, %arg106, %none_883, %628, %629, %630, %false_890, %631, %int1_893 : !torch.vtensor<[1,64,14,14],f32>, !torch.vtensor<[384,64,1,1],f32>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[1,384,14,14],f32>
    %int6_894 = torch.constant.int 6
    %633 = torch.prims.convert_element_type %arg107, %int6_894 : !torch.vtensor<[384],f32>, !torch.int -> !torch.vtensor<[384],f32>
    %int6_895 = torch.constant.int 6
    %634 = torch.prims.convert_element_type %arg108, %int6_895 : !torch.vtensor<[384],f32>, !torch.int -> !torch.vtensor<[384],f32>
    %float1.000000e-03_896 = torch.constant.float 1.000000e-03
    %int1_897 = torch.constant.int 1
    %635 = torch.aten.add.Scalar %634, %float1.000000e-03_896, %int1_897 : !torch.vtensor<[384],f32>, !torch.float, !torch.int -> !torch.vtensor<[384],f32>
    %636 = torch.aten.sqrt %635 : !torch.vtensor<[384],f32> -> !torch.vtensor<[384],f32>
    %637 = torch.aten.reciprocal %636 : !torch.vtensor<[384],f32> -> !torch.vtensor<[384],f32>
    %int1_898 = torch.constant.int 1
    %638 = torch.aten.mul.Scalar %637, %int1_898 : !torch.vtensor<[384],f32>, !torch.int -> !torch.vtensor<[384],f32>
    %int0_899 = torch.constant.int 0
    %639 = torch.prim.ListConstruct %int0_899 : (!torch.int) -> !torch.list<int>
    %none_900 = torch.constant.none
    %none_901 = torch.constant.none
    %none_902 = torch.constant.none
    %false_903 = torch.constant.bool false
    %640 = torch.aten.new_zeros %632, %639, %none_900, %none_901, %none_902, %false_903 : !torch.vtensor<[1,384,14,14],f32>, !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[0],f32>
    %int0_904 = torch.constant.int 0
    %641 = torch.prim.ListConstruct %int0_904 : (!torch.int) -> !torch.list<int>
    %none_905 = torch.constant.none
    %none_906 = torch.constant.none
    %none_907 = torch.constant.none
    %false_908 = torch.constant.bool false
    %642 = torch.aten.new_zeros %632, %641, %none_905, %none_906, %none_907, %false_908 : !torch.vtensor<[1,384,14,14],f32>, !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[0],f32>
    %int-1_909 = torch.constant.int -1
    %643 = torch.aten.unsqueeze %633, %int-1_909 : !torch.vtensor<[384],f32>, !torch.int -> !torch.vtensor<[384,1],f32>
    %int-1_910 = torch.constant.int -1
    %644 = torch.aten.unsqueeze %643, %int-1_910 : !torch.vtensor<[384,1],f32>, !torch.int -> !torch.vtensor<[384,1,1],f32>
    %int-1_911 = torch.constant.int -1
    %645 = torch.aten.unsqueeze %638, %int-1_911 : !torch.vtensor<[384],f32>, !torch.int -> !torch.vtensor<[384,1],f32>
    %int-1_912 = torch.constant.int -1
    %646 = torch.aten.unsqueeze %645, %int-1_912 : !torch.vtensor<[384,1],f32>, !torch.int -> !torch.vtensor<[384,1,1],f32>
    %int1_913 = torch.constant.int 1
    %647 = torch.aten.sub.Tensor %632, %644, %int1_913 : !torch.vtensor<[1,384,14,14],f32>, !torch.vtensor<[384,1,1],f32>, !torch.int -> !torch.vtensor<[1,384,14,14],f32>
    %648 = torch.aten.mul.Tensor %647, %646 : !torch.vtensor<[1,384,14,14],f32>, !torch.vtensor<[384,1,1],f32> -> !torch.vtensor<[1,384,14,14],f32>
    %int-1_914 = torch.constant.int -1
    %649 = torch.aten.unsqueeze %arg109, %int-1_914 : !torch.vtensor<[384],f32>, !torch.int -> !torch.vtensor<[384,1],f32>
    %int-1_915 = torch.constant.int -1
    %650 = torch.aten.unsqueeze %649, %int-1_915 : !torch.vtensor<[384,1],f32>, !torch.int -> !torch.vtensor<[384,1,1],f32>
    %651 = torch.aten.mul.Tensor %648, %650 : !torch.vtensor<[1,384,14,14],f32>, !torch.vtensor<[384,1,1],f32> -> !torch.vtensor<[1,384,14,14],f32>
    %int-1_916 = torch.constant.int -1
    %652 = torch.aten.unsqueeze %arg110, %int-1_916 : !torch.vtensor<[384],f32>, !torch.int -> !torch.vtensor<[384,1],f32>
    %int-1_917 = torch.constant.int -1
    %653 = torch.aten.unsqueeze %652, %int-1_917 : !torch.vtensor<[384,1],f32>, !torch.int -> !torch.vtensor<[384,1,1],f32>
    %int1_918 = torch.constant.int 1
    %654 = torch.aten.add.Tensor %651, %653, %int1_918 : !torch.vtensor<[1,384,14,14],f32>, !torch.vtensor<[384,1,1],f32>, !torch.int -> !torch.vtensor<[1,384,14,14],f32>
    %float0.000000e00_919 = torch.constant.float 0.000000e+00
    %float6.000000e00_920 = torch.constant.float 6.000000e+00
    %655 = torch.aten.hardtanh %654, %float0.000000e00_919, %float6.000000e00_920 : !torch.vtensor<[1,384,14,14],f32>, !torch.float, !torch.float -> !torch.vtensor<[1,384,14,14],f32>
    %int1_921 = torch.constant.int 1
    %int1_922 = torch.constant.int 1
    %int1_923 = torch.constant.int 1
    %int1_924 = torch.constant.int 1
    %656 = torch.prim.ListConstruct %int1_921, %int1_922, %int1_923, %int1_924 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %float0.000000e00_925 = torch.constant.float 0.000000e+00
    %657 = torch.aten.constant_pad_nd %655, %656, %float0.000000e00_925 : !torch.vtensor<[1,384,14,14],f32>, !torch.list<int>, !torch.float -> !torch.vtensor<[1,384,16,16],f32>
    %none_926 = torch.constant.none
    %int1_927 = torch.constant.int 1
    %int1_928 = torch.constant.int 1
    %658 = torch.prim.ListConstruct %int1_927, %int1_928 : (!torch.int, !torch.int) -> !torch.list<int>
    %int0_929 = torch.constant.int 0
    %int0_930 = torch.constant.int 0
    %659 = torch.prim.ListConstruct %int0_929, %int0_930 : (!torch.int, !torch.int) -> !torch.list<int>
    %int1_931 = torch.constant.int 1
    %int1_932 = torch.constant.int 1
    %660 = torch.prim.ListConstruct %int1_931, %int1_932 : (!torch.int, !torch.int) -> !torch.list<int>
    %false_933 = torch.constant.bool false
    %int0_934 = torch.constant.int 0
    %int0_935 = torch.constant.int 0
    %661 = torch.prim.ListConstruct %int0_934, %int0_935 : (!torch.int, !torch.int) -> !torch.list<int>
    %int384 = torch.constant.int 384
    %662 = torch.aten.convolution %657, %arg111, %none_926, %658, %659, %660, %false_933, %661, %int384 : !torch.vtensor<[1,384,16,16],f32>, !torch.vtensor<[384,1,3,3],f32>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[1,384,14,14],f32>
    %int6_936 = torch.constant.int 6
    %663 = torch.prims.convert_element_type %arg112, %int6_936 : !torch.vtensor<[384],f32>, !torch.int -> !torch.vtensor<[384],f32>
    %int6_937 = torch.constant.int 6
    %664 = torch.prims.convert_element_type %arg113, %int6_937 : !torch.vtensor<[384],f32>, !torch.int -> !torch.vtensor<[384],f32>
    %float1.000000e-03_938 = torch.constant.float 1.000000e-03
    %int1_939 = torch.constant.int 1
    %665 = torch.aten.add.Scalar %664, %float1.000000e-03_938, %int1_939 : !torch.vtensor<[384],f32>, !torch.float, !torch.int -> !torch.vtensor<[384],f32>
    %666 = torch.aten.sqrt %665 : !torch.vtensor<[384],f32> -> !torch.vtensor<[384],f32>
    %667 = torch.aten.reciprocal %666 : !torch.vtensor<[384],f32> -> !torch.vtensor<[384],f32>
    %int1_940 = torch.constant.int 1
    %668 = torch.aten.mul.Scalar %667, %int1_940 : !torch.vtensor<[384],f32>, !torch.int -> !torch.vtensor<[384],f32>
    %int0_941 = torch.constant.int 0
    %669 = torch.prim.ListConstruct %int0_941 : (!torch.int) -> !torch.list<int>
    %none_942 = torch.constant.none
    %none_943 = torch.constant.none
    %none_944 = torch.constant.none
    %false_945 = torch.constant.bool false
    %670 = torch.aten.new_zeros %662, %669, %none_942, %none_943, %none_944, %false_945 : !torch.vtensor<[1,384,14,14],f32>, !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[0],f32>
    %int0_946 = torch.constant.int 0
    %671 = torch.prim.ListConstruct %int0_946 : (!torch.int) -> !torch.list<int>
    %none_947 = torch.constant.none
    %none_948 = torch.constant.none
    %none_949 = torch.constant.none
    %false_950 = torch.constant.bool false
    %672 = torch.aten.new_zeros %662, %671, %none_947, %none_948, %none_949, %false_950 : !torch.vtensor<[1,384,14,14],f32>, !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[0],f32>
    %int-1_951 = torch.constant.int -1
    %673 = torch.aten.unsqueeze %663, %int-1_951 : !torch.vtensor<[384],f32>, !torch.int -> !torch.vtensor<[384,1],f32>
    %int-1_952 = torch.constant.int -1
    %674 = torch.aten.unsqueeze %673, %int-1_952 : !torch.vtensor<[384,1],f32>, !torch.int -> !torch.vtensor<[384,1,1],f32>
    %int-1_953 = torch.constant.int -1
    %675 = torch.aten.unsqueeze %668, %int-1_953 : !torch.vtensor<[384],f32>, !torch.int -> !torch.vtensor<[384,1],f32>
    %int-1_954 = torch.constant.int -1
    %676 = torch.aten.unsqueeze %675, %int-1_954 : !torch.vtensor<[384,1],f32>, !torch.int -> !torch.vtensor<[384,1,1],f32>
    %int1_955 = torch.constant.int 1
    %677 = torch.aten.sub.Tensor %662, %674, %int1_955 : !torch.vtensor<[1,384,14,14],f32>, !torch.vtensor<[384,1,1],f32>, !torch.int -> !torch.vtensor<[1,384,14,14],f32>
    %678 = torch.aten.mul.Tensor %677, %676 : !torch.vtensor<[1,384,14,14],f32>, !torch.vtensor<[384,1,1],f32> -> !torch.vtensor<[1,384,14,14],f32>
    %int-1_956 = torch.constant.int -1
    %679 = torch.aten.unsqueeze %arg114, %int-1_956 : !torch.vtensor<[384],f32>, !torch.int -> !torch.vtensor<[384,1],f32>
    %int-1_957 = torch.constant.int -1
    %680 = torch.aten.unsqueeze %679, %int-1_957 : !torch.vtensor<[384,1],f32>, !torch.int -> !torch.vtensor<[384,1,1],f32>
    %681 = torch.aten.mul.Tensor %678, %680 : !torch.vtensor<[1,384,14,14],f32>, !torch.vtensor<[384,1,1],f32> -> !torch.vtensor<[1,384,14,14],f32>
    %int-1_958 = torch.constant.int -1
    %682 = torch.aten.unsqueeze %arg115, %int-1_958 : !torch.vtensor<[384],f32>, !torch.int -> !torch.vtensor<[384,1],f32>
    %int-1_959 = torch.constant.int -1
    %683 = torch.aten.unsqueeze %682, %int-1_959 : !torch.vtensor<[384,1],f32>, !torch.int -> !torch.vtensor<[384,1,1],f32>
    %int1_960 = torch.constant.int 1
    %684 = torch.aten.add.Tensor %681, %683, %int1_960 : !torch.vtensor<[1,384,14,14],f32>, !torch.vtensor<[384,1,1],f32>, !torch.int -> !torch.vtensor<[1,384,14,14],f32>
    %float0.000000e00_961 = torch.constant.float 0.000000e+00
    %float6.000000e00_962 = torch.constant.float 6.000000e+00
    %685 = torch.aten.hardtanh %684, %float0.000000e00_961, %float6.000000e00_962 : !torch.vtensor<[1,384,14,14],f32>, !torch.float, !torch.float -> !torch.vtensor<[1,384,14,14],f32>
    %int0_963 = torch.constant.int 0
    %int0_964 = torch.constant.int 0
    %int0_965 = torch.constant.int 0
    %int0_966 = torch.constant.int 0
    %686 = torch.prim.ListConstruct %int0_963, %int0_964, %int0_965, %int0_966 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %float0.000000e00_967 = torch.constant.float 0.000000e+00
    %687 = torch.aten.constant_pad_nd %685, %686, %float0.000000e00_967 : !torch.vtensor<[1,384,14,14],f32>, !torch.list<int>, !torch.float -> !torch.vtensor<[1,384,14,14],f32>
    %none_968 = torch.constant.none
    %int1_969 = torch.constant.int 1
    %int1_970 = torch.constant.int 1
    %688 = torch.prim.ListConstruct %int1_969, %int1_970 : (!torch.int, !torch.int) -> !torch.list<int>
    %int0_971 = torch.constant.int 0
    %int0_972 = torch.constant.int 0
    %689 = torch.prim.ListConstruct %int0_971, %int0_972 : (!torch.int, !torch.int) -> !torch.list<int>
    %int1_973 = torch.constant.int 1
    %int1_974 = torch.constant.int 1
    %690 = torch.prim.ListConstruct %int1_973, %int1_974 : (!torch.int, !torch.int) -> !torch.list<int>
    %false_975 = torch.constant.bool false
    %int0_976 = torch.constant.int 0
    %int0_977 = torch.constant.int 0
    %691 = torch.prim.ListConstruct %int0_976, %int0_977 : (!torch.int, !torch.int) -> !torch.list<int>
    %int1_978 = torch.constant.int 1
    %692 = torch.aten.convolution %687, %arg116, %none_968, %688, %689, %690, %false_975, %691, %int1_978 : !torch.vtensor<[1,384,14,14],f32>, !torch.vtensor<[64,384,1,1],f32>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[1,64,14,14],f32>
    %int6_979 = torch.constant.int 6
    %693 = torch.prims.convert_element_type %arg117, %int6_979 : !torch.vtensor<[64],f32>, !torch.int -> !torch.vtensor<[64],f32>
    %int6_980 = torch.constant.int 6
    %694 = torch.prims.convert_element_type %arg118, %int6_980 : !torch.vtensor<[64],f32>, !torch.int -> !torch.vtensor<[64],f32>
    %float1.000000e-03_981 = torch.constant.float 1.000000e-03
    %int1_982 = torch.constant.int 1
    %695 = torch.aten.add.Scalar %694, %float1.000000e-03_981, %int1_982 : !torch.vtensor<[64],f32>, !torch.float, !torch.int -> !torch.vtensor<[64],f32>
    %696 = torch.aten.sqrt %695 : !torch.vtensor<[64],f32> -> !torch.vtensor<[64],f32>
    %697 = torch.aten.reciprocal %696 : !torch.vtensor<[64],f32> -> !torch.vtensor<[64],f32>
    %int1_983 = torch.constant.int 1
    %698 = torch.aten.mul.Scalar %697, %int1_983 : !torch.vtensor<[64],f32>, !torch.int -> !torch.vtensor<[64],f32>
    %int0_984 = torch.constant.int 0
    %699 = torch.prim.ListConstruct %int0_984 : (!torch.int) -> !torch.list<int>
    %none_985 = torch.constant.none
    %none_986 = torch.constant.none
    %none_987 = torch.constant.none
    %false_988 = torch.constant.bool false
    %700 = torch.aten.new_zeros %692, %699, %none_985, %none_986, %none_987, %false_988 : !torch.vtensor<[1,64,14,14],f32>, !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[0],f32>
    %int0_989 = torch.constant.int 0
    %701 = torch.prim.ListConstruct %int0_989 : (!torch.int) -> !torch.list<int>
    %none_990 = torch.constant.none
    %none_991 = torch.constant.none
    %none_992 = torch.constant.none
    %false_993 = torch.constant.bool false
    %702 = torch.aten.new_zeros %692, %701, %none_990, %none_991, %none_992, %false_993 : !torch.vtensor<[1,64,14,14],f32>, !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[0],f32>
    %int-1_994 = torch.constant.int -1
    %703 = torch.aten.unsqueeze %693, %int-1_994 : !torch.vtensor<[64],f32>, !torch.int -> !torch.vtensor<[64,1],f32>
    %int-1_995 = torch.constant.int -1
    %704 = torch.aten.unsqueeze %703, %int-1_995 : !torch.vtensor<[64,1],f32>, !torch.int -> !torch.vtensor<[64,1,1],f32>
    %int-1_996 = torch.constant.int -1
    %705 = torch.aten.unsqueeze %698, %int-1_996 : !torch.vtensor<[64],f32>, !torch.int -> !torch.vtensor<[64,1],f32>
    %int-1_997 = torch.constant.int -1
    %706 = torch.aten.unsqueeze %705, %int-1_997 : !torch.vtensor<[64,1],f32>, !torch.int -> !torch.vtensor<[64,1,1],f32>
    %int1_998 = torch.constant.int 1
    %707 = torch.aten.sub.Tensor %692, %704, %int1_998 : !torch.vtensor<[1,64,14,14],f32>, !torch.vtensor<[64,1,1],f32>, !torch.int -> !torch.vtensor<[1,64,14,14],f32>
    %708 = torch.aten.mul.Tensor %707, %706 : !torch.vtensor<[1,64,14,14],f32>, !torch.vtensor<[64,1,1],f32> -> !torch.vtensor<[1,64,14,14],f32>
    %int-1_999 = torch.constant.int -1
    %709 = torch.aten.unsqueeze %arg119, %int-1_999 : !torch.vtensor<[64],f32>, !torch.int -> !torch.vtensor<[64,1],f32>
    %int-1_1000 = torch.constant.int -1
    %710 = torch.aten.unsqueeze %709, %int-1_1000 : !torch.vtensor<[64,1],f32>, !torch.int -> !torch.vtensor<[64,1,1],f32>
    %711 = torch.aten.mul.Tensor %708, %710 : !torch.vtensor<[1,64,14,14],f32>, !torch.vtensor<[64,1,1],f32> -> !torch.vtensor<[1,64,14,14],f32>
    %int-1_1001 = torch.constant.int -1
    %712 = torch.aten.unsqueeze %arg120, %int-1_1001 : !torch.vtensor<[64],f32>, !torch.int -> !torch.vtensor<[64,1],f32>
    %int-1_1002 = torch.constant.int -1
    %713 = torch.aten.unsqueeze %712, %int-1_1002 : !torch.vtensor<[64,1],f32>, !torch.int -> !torch.vtensor<[64,1,1],f32>
    %int1_1003 = torch.constant.int 1
    %714 = torch.aten.add.Tensor %711, %713, %int1_1003 : !torch.vtensor<[1,64,14,14],f32>, !torch.vtensor<[64,1,1],f32>, !torch.int -> !torch.vtensor<[1,64,14,14],f32>
    %int1_1004 = torch.constant.int 1
    %715 = torch.aten.add.Tensor %625, %714, %int1_1004 : !torch.vtensor<[1,64,14,14],f32>, !torch.vtensor<[1,64,14,14],f32>, !torch.int -> !torch.vtensor<[1,64,14,14],f32>
    %int0_1005 = torch.constant.int 0
    %int0_1006 = torch.constant.int 0
    %int0_1007 = torch.constant.int 0
    %int0_1008 = torch.constant.int 0
    %716 = torch.prim.ListConstruct %int0_1005, %int0_1006, %int0_1007, %int0_1008 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %float0.000000e00_1009 = torch.constant.float 0.000000e+00
    %717 = torch.aten.constant_pad_nd %715, %716, %float0.000000e00_1009 : !torch.vtensor<[1,64,14,14],f32>, !torch.list<int>, !torch.float -> !torch.vtensor<[1,64,14,14],f32>
    %none_1010 = torch.constant.none
    %int1_1011 = torch.constant.int 1
    %int1_1012 = torch.constant.int 1
    %718 = torch.prim.ListConstruct %int1_1011, %int1_1012 : (!torch.int, !torch.int) -> !torch.list<int>
    %int0_1013 = torch.constant.int 0
    %int0_1014 = torch.constant.int 0
    %719 = torch.prim.ListConstruct %int0_1013, %int0_1014 : (!torch.int, !torch.int) -> !torch.list<int>
    %int1_1015 = torch.constant.int 1
    %int1_1016 = torch.constant.int 1
    %720 = torch.prim.ListConstruct %int1_1015, %int1_1016 : (!torch.int, !torch.int) -> !torch.list<int>
    %false_1017 = torch.constant.bool false
    %int0_1018 = torch.constant.int 0
    %int0_1019 = torch.constant.int 0
    %721 = torch.prim.ListConstruct %int0_1018, %int0_1019 : (!torch.int, !torch.int) -> !torch.list<int>
    %int1_1020 = torch.constant.int 1
    %722 = torch.aten.convolution %717, %arg121, %none_1010, %718, %719, %720, %false_1017, %721, %int1_1020 : !torch.vtensor<[1,64,14,14],f32>, !torch.vtensor<[384,64,1,1],f32>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[1,384,14,14],f32>
    %int6_1021 = torch.constant.int 6
    %723 = torch.prims.convert_element_type %arg122, %int6_1021 : !torch.vtensor<[384],f32>, !torch.int -> !torch.vtensor<[384],f32>
    %int6_1022 = torch.constant.int 6
    %724 = torch.prims.convert_element_type %arg123, %int6_1022 : !torch.vtensor<[384],f32>, !torch.int -> !torch.vtensor<[384],f32>
    %float1.000000e-03_1023 = torch.constant.float 1.000000e-03
    %int1_1024 = torch.constant.int 1
    %725 = torch.aten.add.Scalar %724, %float1.000000e-03_1023, %int1_1024 : !torch.vtensor<[384],f32>, !torch.float, !torch.int -> !torch.vtensor<[384],f32>
    %726 = torch.aten.sqrt %725 : !torch.vtensor<[384],f32> -> !torch.vtensor<[384],f32>
    %727 = torch.aten.reciprocal %726 : !torch.vtensor<[384],f32> -> !torch.vtensor<[384],f32>
    %int1_1025 = torch.constant.int 1
    %728 = torch.aten.mul.Scalar %727, %int1_1025 : !torch.vtensor<[384],f32>, !torch.int -> !torch.vtensor<[384],f32>
    %int0_1026 = torch.constant.int 0
    %729 = torch.prim.ListConstruct %int0_1026 : (!torch.int) -> !torch.list<int>
    %none_1027 = torch.constant.none
    %none_1028 = torch.constant.none
    %none_1029 = torch.constant.none
    %false_1030 = torch.constant.bool false
    %730 = torch.aten.new_zeros %722, %729, %none_1027, %none_1028, %none_1029, %false_1030 : !torch.vtensor<[1,384,14,14],f32>, !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[0],f32>
    %int0_1031 = torch.constant.int 0
    %731 = torch.prim.ListConstruct %int0_1031 : (!torch.int) -> !torch.list<int>
    %none_1032 = torch.constant.none
    %none_1033 = torch.constant.none
    %none_1034 = torch.constant.none
    %false_1035 = torch.constant.bool false
    %732 = torch.aten.new_zeros %722, %731, %none_1032, %none_1033, %none_1034, %false_1035 : !torch.vtensor<[1,384,14,14],f32>, !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[0],f32>
    %int-1_1036 = torch.constant.int -1
    %733 = torch.aten.unsqueeze %723, %int-1_1036 : !torch.vtensor<[384],f32>, !torch.int -> !torch.vtensor<[384,1],f32>
    %int-1_1037 = torch.constant.int -1
    %734 = torch.aten.unsqueeze %733, %int-1_1037 : !torch.vtensor<[384,1],f32>, !torch.int -> !torch.vtensor<[384,1,1],f32>
    %int-1_1038 = torch.constant.int -1
    %735 = torch.aten.unsqueeze %728, %int-1_1038 : !torch.vtensor<[384],f32>, !torch.int -> !torch.vtensor<[384,1],f32>
    %int-1_1039 = torch.constant.int -1
    %736 = torch.aten.unsqueeze %735, %int-1_1039 : !torch.vtensor<[384,1],f32>, !torch.int -> !torch.vtensor<[384,1,1],f32>
    %int1_1040 = torch.constant.int 1
    %737 = torch.aten.sub.Tensor %722, %734, %int1_1040 : !torch.vtensor<[1,384,14,14],f32>, !torch.vtensor<[384,1,1],f32>, !torch.int -> !torch.vtensor<[1,384,14,14],f32>
    %738 = torch.aten.mul.Tensor %737, %736 : !torch.vtensor<[1,384,14,14],f32>, !torch.vtensor<[384,1,1],f32> -> !torch.vtensor<[1,384,14,14],f32>
    %int-1_1041 = torch.constant.int -1
    %739 = torch.aten.unsqueeze %arg124, %int-1_1041 : !torch.vtensor<[384],f32>, !torch.int -> !torch.vtensor<[384,1],f32>
    %int-1_1042 = torch.constant.int -1
    %740 = torch.aten.unsqueeze %739, %int-1_1042 : !torch.vtensor<[384,1],f32>, !torch.int -> !torch.vtensor<[384,1,1],f32>
    %741 = torch.aten.mul.Tensor %738, %740 : !torch.vtensor<[1,384,14,14],f32>, !torch.vtensor<[384,1,1],f32> -> !torch.vtensor<[1,384,14,14],f32>
    %int-1_1043 = torch.constant.int -1
    %742 = torch.aten.unsqueeze %arg125, %int-1_1043 : !torch.vtensor<[384],f32>, !torch.int -> !torch.vtensor<[384,1],f32>
    %int-1_1044 = torch.constant.int -1
    %743 = torch.aten.unsqueeze %742, %int-1_1044 : !torch.vtensor<[384,1],f32>, !torch.int -> !torch.vtensor<[384,1,1],f32>
    %int1_1045 = torch.constant.int 1
    %744 = torch.aten.add.Tensor %741, %743, %int1_1045 : !torch.vtensor<[1,384,14,14],f32>, !torch.vtensor<[384,1,1],f32>, !torch.int -> !torch.vtensor<[1,384,14,14],f32>
    %float0.000000e00_1046 = torch.constant.float 0.000000e+00
    %float6.000000e00_1047 = torch.constant.float 6.000000e+00
    %745 = torch.aten.hardtanh %744, %float0.000000e00_1046, %float6.000000e00_1047 : !torch.vtensor<[1,384,14,14],f32>, !torch.float, !torch.float -> !torch.vtensor<[1,384,14,14],f32>
    %int1_1048 = torch.constant.int 1
    %int1_1049 = torch.constant.int 1
    %int1_1050 = torch.constant.int 1
    %int1_1051 = torch.constant.int 1
    %746 = torch.prim.ListConstruct %int1_1048, %int1_1049, %int1_1050, %int1_1051 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %float0.000000e00_1052 = torch.constant.float 0.000000e+00
    %747 = torch.aten.constant_pad_nd %745, %746, %float0.000000e00_1052 : !torch.vtensor<[1,384,14,14],f32>, !torch.list<int>, !torch.float -> !torch.vtensor<[1,384,16,16],f32>
    %none_1053 = torch.constant.none
    %int1_1054 = torch.constant.int 1
    %int1_1055 = torch.constant.int 1
    %748 = torch.prim.ListConstruct %int1_1054, %int1_1055 : (!torch.int, !torch.int) -> !torch.list<int>
    %int0_1056 = torch.constant.int 0
    %int0_1057 = torch.constant.int 0
    %749 = torch.prim.ListConstruct %int0_1056, %int0_1057 : (!torch.int, !torch.int) -> !torch.list<int>
    %int1_1058 = torch.constant.int 1
    %int1_1059 = torch.constant.int 1
    %750 = torch.prim.ListConstruct %int1_1058, %int1_1059 : (!torch.int, !torch.int) -> !torch.list<int>
    %false_1060 = torch.constant.bool false
    %int0_1061 = torch.constant.int 0
    %int0_1062 = torch.constant.int 0
    %751 = torch.prim.ListConstruct %int0_1061, %int0_1062 : (!torch.int, !torch.int) -> !torch.list<int>
    %int384_1063 = torch.constant.int 384
    %752 = torch.aten.convolution %747, %arg126, %none_1053, %748, %749, %750, %false_1060, %751, %int384_1063 : !torch.vtensor<[1,384,16,16],f32>, !torch.vtensor<[384,1,3,3],f32>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[1,384,14,14],f32>
    %int6_1064 = torch.constant.int 6
    %753 = torch.prims.convert_element_type %arg127, %int6_1064 : !torch.vtensor<[384],f32>, !torch.int -> !torch.vtensor<[384],f32>
    %int6_1065 = torch.constant.int 6
    %754 = torch.prims.convert_element_type %arg128, %int6_1065 : !torch.vtensor<[384],f32>, !torch.int -> !torch.vtensor<[384],f32>
    %float1.000000e-03_1066 = torch.constant.float 1.000000e-03
    %int1_1067 = torch.constant.int 1
    %755 = torch.aten.add.Scalar %754, %float1.000000e-03_1066, %int1_1067 : !torch.vtensor<[384],f32>, !torch.float, !torch.int -> !torch.vtensor<[384],f32>
    %756 = torch.aten.sqrt %755 : !torch.vtensor<[384],f32> -> !torch.vtensor<[384],f32>
    %757 = torch.aten.reciprocal %756 : !torch.vtensor<[384],f32> -> !torch.vtensor<[384],f32>
    %int1_1068 = torch.constant.int 1
    %758 = torch.aten.mul.Scalar %757, %int1_1068 : !torch.vtensor<[384],f32>, !torch.int -> !torch.vtensor<[384],f32>
    %int0_1069 = torch.constant.int 0
    %759 = torch.prim.ListConstruct %int0_1069 : (!torch.int) -> !torch.list<int>
    %none_1070 = torch.constant.none
    %none_1071 = torch.constant.none
    %none_1072 = torch.constant.none
    %false_1073 = torch.constant.bool false
    %760 = torch.aten.new_zeros %752, %759, %none_1070, %none_1071, %none_1072, %false_1073 : !torch.vtensor<[1,384,14,14],f32>, !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[0],f32>
    %int0_1074 = torch.constant.int 0
    %761 = torch.prim.ListConstruct %int0_1074 : (!torch.int) -> !torch.list<int>
    %none_1075 = torch.constant.none
    %none_1076 = torch.constant.none
    %none_1077 = torch.constant.none
    %false_1078 = torch.constant.bool false
    %762 = torch.aten.new_zeros %752, %761, %none_1075, %none_1076, %none_1077, %false_1078 : !torch.vtensor<[1,384,14,14],f32>, !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[0],f32>
    %int-1_1079 = torch.constant.int -1
    %763 = torch.aten.unsqueeze %753, %int-1_1079 : !torch.vtensor<[384],f32>, !torch.int -> !torch.vtensor<[384,1],f32>
    %int-1_1080 = torch.constant.int -1
    %764 = torch.aten.unsqueeze %763, %int-1_1080 : !torch.vtensor<[384,1],f32>, !torch.int -> !torch.vtensor<[384,1,1],f32>
    %int-1_1081 = torch.constant.int -1
    %765 = torch.aten.unsqueeze %758, %int-1_1081 : !torch.vtensor<[384],f32>, !torch.int -> !torch.vtensor<[384,1],f32>
    %int-1_1082 = torch.constant.int -1
    %766 = torch.aten.unsqueeze %765, %int-1_1082 : !torch.vtensor<[384,1],f32>, !torch.int -> !torch.vtensor<[384,1,1],f32>
    %int1_1083 = torch.constant.int 1
    %767 = torch.aten.sub.Tensor %752, %764, %int1_1083 : !torch.vtensor<[1,384,14,14],f32>, !torch.vtensor<[384,1,1],f32>, !torch.int -> !torch.vtensor<[1,384,14,14],f32>
    %768 = torch.aten.mul.Tensor %767, %766 : !torch.vtensor<[1,384,14,14],f32>, !torch.vtensor<[384,1,1],f32> -> !torch.vtensor<[1,384,14,14],f32>
    %int-1_1084 = torch.constant.int -1
    %769 = torch.aten.unsqueeze %arg129, %int-1_1084 : !torch.vtensor<[384],f32>, !torch.int -> !torch.vtensor<[384,1],f32>
    %int-1_1085 = torch.constant.int -1
    %770 = torch.aten.unsqueeze %769, %int-1_1085 : !torch.vtensor<[384,1],f32>, !torch.int -> !torch.vtensor<[384,1,1],f32>
    %771 = torch.aten.mul.Tensor %768, %770 : !torch.vtensor<[1,384,14,14],f32>, !torch.vtensor<[384,1,1],f32> -> !torch.vtensor<[1,384,14,14],f32>
    %int-1_1086 = torch.constant.int -1
    %772 = torch.aten.unsqueeze %arg130, %int-1_1086 : !torch.vtensor<[384],f32>, !torch.int -> !torch.vtensor<[384,1],f32>
    %int-1_1087 = torch.constant.int -1
    %773 = torch.aten.unsqueeze %772, %int-1_1087 : !torch.vtensor<[384,1],f32>, !torch.int -> !torch.vtensor<[384,1,1],f32>
    %int1_1088 = torch.constant.int 1
    %774 = torch.aten.add.Tensor %771, %773, %int1_1088 : !torch.vtensor<[1,384,14,14],f32>, !torch.vtensor<[384,1,1],f32>, !torch.int -> !torch.vtensor<[1,384,14,14],f32>
    %float0.000000e00_1089 = torch.constant.float 0.000000e+00
    %float6.000000e00_1090 = torch.constant.float 6.000000e+00
    %775 = torch.aten.hardtanh %774, %float0.000000e00_1089, %float6.000000e00_1090 : !torch.vtensor<[1,384,14,14],f32>, !torch.float, !torch.float -> !torch.vtensor<[1,384,14,14],f32>
    %int0_1091 = torch.constant.int 0
    %int0_1092 = torch.constant.int 0
    %int0_1093 = torch.constant.int 0
    %int0_1094 = torch.constant.int 0
    %776 = torch.prim.ListConstruct %int0_1091, %int0_1092, %int0_1093, %int0_1094 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %float0.000000e00_1095 = torch.constant.float 0.000000e+00
    %777 = torch.aten.constant_pad_nd %775, %776, %float0.000000e00_1095 : !torch.vtensor<[1,384,14,14],f32>, !torch.list<int>, !torch.float -> !torch.vtensor<[1,384,14,14],f32>
    %none_1096 = torch.constant.none
    %int1_1097 = torch.constant.int 1
    %int1_1098 = torch.constant.int 1
    %778 = torch.prim.ListConstruct %int1_1097, %int1_1098 : (!torch.int, !torch.int) -> !torch.list<int>
    %int0_1099 = torch.constant.int 0
    %int0_1100 = torch.constant.int 0
    %779 = torch.prim.ListConstruct %int0_1099, %int0_1100 : (!torch.int, !torch.int) -> !torch.list<int>
    %int1_1101 = torch.constant.int 1
    %int1_1102 = torch.constant.int 1
    %780 = torch.prim.ListConstruct %int1_1101, %int1_1102 : (!torch.int, !torch.int) -> !torch.list<int>
    %false_1103 = torch.constant.bool false
    %int0_1104 = torch.constant.int 0
    %int0_1105 = torch.constant.int 0
    %781 = torch.prim.ListConstruct %int0_1104, %int0_1105 : (!torch.int, !torch.int) -> !torch.list<int>
    %int1_1106 = torch.constant.int 1
    %782 = torch.aten.convolution %777, %arg131, %none_1096, %778, %779, %780, %false_1103, %781, %int1_1106 : !torch.vtensor<[1,384,14,14],f32>, !torch.vtensor<[64,384,1,1],f32>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[1,64,14,14],f32>
    %int6_1107 = torch.constant.int 6
    %783 = torch.prims.convert_element_type %arg132, %int6_1107 : !torch.vtensor<[64],f32>, !torch.int -> !torch.vtensor<[64],f32>
    %int6_1108 = torch.constant.int 6
    %784 = torch.prims.convert_element_type %arg133, %int6_1108 : !torch.vtensor<[64],f32>, !torch.int -> !torch.vtensor<[64],f32>
    %float1.000000e-03_1109 = torch.constant.float 1.000000e-03
    %int1_1110 = torch.constant.int 1
    %785 = torch.aten.add.Scalar %784, %float1.000000e-03_1109, %int1_1110 : !torch.vtensor<[64],f32>, !torch.float, !torch.int -> !torch.vtensor<[64],f32>
    %786 = torch.aten.sqrt %785 : !torch.vtensor<[64],f32> -> !torch.vtensor<[64],f32>
    %787 = torch.aten.reciprocal %786 : !torch.vtensor<[64],f32> -> !torch.vtensor<[64],f32>
    %int1_1111 = torch.constant.int 1
    %788 = torch.aten.mul.Scalar %787, %int1_1111 : !torch.vtensor<[64],f32>, !torch.int -> !torch.vtensor<[64],f32>
    %int0_1112 = torch.constant.int 0
    %789 = torch.prim.ListConstruct %int0_1112 : (!torch.int) -> !torch.list<int>
    %none_1113 = torch.constant.none
    %none_1114 = torch.constant.none
    %none_1115 = torch.constant.none
    %false_1116 = torch.constant.bool false
    %790 = torch.aten.new_zeros %782, %789, %none_1113, %none_1114, %none_1115, %false_1116 : !torch.vtensor<[1,64,14,14],f32>, !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[0],f32>
    %int0_1117 = torch.constant.int 0
    %791 = torch.prim.ListConstruct %int0_1117 : (!torch.int) -> !torch.list<int>
    %none_1118 = torch.constant.none
    %none_1119 = torch.constant.none
    %none_1120 = torch.constant.none
    %false_1121 = torch.constant.bool false
    %792 = torch.aten.new_zeros %782, %791, %none_1118, %none_1119, %none_1120, %false_1121 : !torch.vtensor<[1,64,14,14],f32>, !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[0],f32>
    %int-1_1122 = torch.constant.int -1
    %793 = torch.aten.unsqueeze %783, %int-1_1122 : !torch.vtensor<[64],f32>, !torch.int -> !torch.vtensor<[64,1],f32>
    %int-1_1123 = torch.constant.int -1
    %794 = torch.aten.unsqueeze %793, %int-1_1123 : !torch.vtensor<[64,1],f32>, !torch.int -> !torch.vtensor<[64,1,1],f32>
    %int-1_1124 = torch.constant.int -1
    %795 = torch.aten.unsqueeze %788, %int-1_1124 : !torch.vtensor<[64],f32>, !torch.int -> !torch.vtensor<[64,1],f32>
    %int-1_1125 = torch.constant.int -1
    %796 = torch.aten.unsqueeze %795, %int-1_1125 : !torch.vtensor<[64,1],f32>, !torch.int -> !torch.vtensor<[64,1,1],f32>
    %int1_1126 = torch.constant.int 1
    %797 = torch.aten.sub.Tensor %782, %794, %int1_1126 : !torch.vtensor<[1,64,14,14],f32>, !torch.vtensor<[64,1,1],f32>, !torch.int -> !torch.vtensor<[1,64,14,14],f32>
    %798 = torch.aten.mul.Tensor %797, %796 : !torch.vtensor<[1,64,14,14],f32>, !torch.vtensor<[64,1,1],f32> -> !torch.vtensor<[1,64,14,14],f32>
    %int-1_1127 = torch.constant.int -1
    %799 = torch.aten.unsqueeze %arg134, %int-1_1127 : !torch.vtensor<[64],f32>, !torch.int -> !torch.vtensor<[64,1],f32>
    %int-1_1128 = torch.constant.int -1
    %800 = torch.aten.unsqueeze %799, %int-1_1128 : !torch.vtensor<[64,1],f32>, !torch.int -> !torch.vtensor<[64,1,1],f32>
    %801 = torch.aten.mul.Tensor %798, %800 : !torch.vtensor<[1,64,14,14],f32>, !torch.vtensor<[64,1,1],f32> -> !torch.vtensor<[1,64,14,14],f32>
    %int-1_1129 = torch.constant.int -1
    %802 = torch.aten.unsqueeze %arg135, %int-1_1129 : !torch.vtensor<[64],f32>, !torch.int -> !torch.vtensor<[64,1],f32>
    %int-1_1130 = torch.constant.int -1
    %803 = torch.aten.unsqueeze %802, %int-1_1130 : !torch.vtensor<[64,1],f32>, !torch.int -> !torch.vtensor<[64,1,1],f32>
    %int1_1131 = torch.constant.int 1
    %804 = torch.aten.add.Tensor %801, %803, %int1_1131 : !torch.vtensor<[1,64,14,14],f32>, !torch.vtensor<[64,1,1],f32>, !torch.int -> !torch.vtensor<[1,64,14,14],f32>
    %int1_1132 = torch.constant.int 1
    %805 = torch.aten.add.Tensor %715, %804, %int1_1132 : !torch.vtensor<[1,64,14,14],f32>, !torch.vtensor<[1,64,14,14],f32>, !torch.int -> !torch.vtensor<[1,64,14,14],f32>
    %int0_1133 = torch.constant.int 0
    %int0_1134 = torch.constant.int 0
    %int0_1135 = torch.constant.int 0
    %int0_1136 = torch.constant.int 0
    %806 = torch.prim.ListConstruct %int0_1133, %int0_1134, %int0_1135, %int0_1136 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %float0.000000e00_1137 = torch.constant.float 0.000000e+00
    %807 = torch.aten.constant_pad_nd %805, %806, %float0.000000e00_1137 : !torch.vtensor<[1,64,14,14],f32>, !torch.list<int>, !torch.float -> !torch.vtensor<[1,64,14,14],f32>
    %none_1138 = torch.constant.none
    %int1_1139 = torch.constant.int 1
    %int1_1140 = torch.constant.int 1
    %808 = torch.prim.ListConstruct %int1_1139, %int1_1140 : (!torch.int, !torch.int) -> !torch.list<int>
    %int0_1141 = torch.constant.int 0
    %int0_1142 = torch.constant.int 0
    %809 = torch.prim.ListConstruct %int0_1141, %int0_1142 : (!torch.int, !torch.int) -> !torch.list<int>
    %int1_1143 = torch.constant.int 1
    %int1_1144 = torch.constant.int 1
    %810 = torch.prim.ListConstruct %int1_1143, %int1_1144 : (!torch.int, !torch.int) -> !torch.list<int>
    %false_1145 = torch.constant.bool false
    %int0_1146 = torch.constant.int 0
    %int0_1147 = torch.constant.int 0
    %811 = torch.prim.ListConstruct %int0_1146, %int0_1147 : (!torch.int, !torch.int) -> !torch.list<int>
    %int1_1148 = torch.constant.int 1
    %812 = torch.aten.convolution %807, %arg136, %none_1138, %808, %809, %810, %false_1145, %811, %int1_1148 : !torch.vtensor<[1,64,14,14],f32>, !torch.vtensor<[384,64,1,1],f32>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[1,384,14,14],f32>
    %int6_1149 = torch.constant.int 6
    %813 = torch.prims.convert_element_type %arg137, %int6_1149 : !torch.vtensor<[384],f32>, !torch.int -> !torch.vtensor<[384],f32>
    %int6_1150 = torch.constant.int 6
    %814 = torch.prims.convert_element_type %arg138, %int6_1150 : !torch.vtensor<[384],f32>, !torch.int -> !torch.vtensor<[384],f32>
    %float1.000000e-03_1151 = torch.constant.float 1.000000e-03
    %int1_1152 = torch.constant.int 1
    %815 = torch.aten.add.Scalar %814, %float1.000000e-03_1151, %int1_1152 : !torch.vtensor<[384],f32>, !torch.float, !torch.int -> !torch.vtensor<[384],f32>
    %816 = torch.aten.sqrt %815 : !torch.vtensor<[384],f32> -> !torch.vtensor<[384],f32>
    %817 = torch.aten.reciprocal %816 : !torch.vtensor<[384],f32> -> !torch.vtensor<[384],f32>
    %int1_1153 = torch.constant.int 1
    %818 = torch.aten.mul.Scalar %817, %int1_1153 : !torch.vtensor<[384],f32>, !torch.int -> !torch.vtensor<[384],f32>
    %int0_1154 = torch.constant.int 0
    %819 = torch.prim.ListConstruct %int0_1154 : (!torch.int) -> !torch.list<int>
    %none_1155 = torch.constant.none
    %none_1156 = torch.constant.none
    %none_1157 = torch.constant.none
    %false_1158 = torch.constant.bool false
    %820 = torch.aten.new_zeros %812, %819, %none_1155, %none_1156, %none_1157, %false_1158 : !torch.vtensor<[1,384,14,14],f32>, !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[0],f32>
    %int0_1159 = torch.constant.int 0
    %821 = torch.prim.ListConstruct %int0_1159 : (!torch.int) -> !torch.list<int>
    %none_1160 = torch.constant.none
    %none_1161 = torch.constant.none
    %none_1162 = torch.constant.none
    %false_1163 = torch.constant.bool false
    %822 = torch.aten.new_zeros %812, %821, %none_1160, %none_1161, %none_1162, %false_1163 : !torch.vtensor<[1,384,14,14],f32>, !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[0],f32>
    %int-1_1164 = torch.constant.int -1
    %823 = torch.aten.unsqueeze %813, %int-1_1164 : !torch.vtensor<[384],f32>, !torch.int -> !torch.vtensor<[384,1],f32>
    %int-1_1165 = torch.constant.int -1
    %824 = torch.aten.unsqueeze %823, %int-1_1165 : !torch.vtensor<[384,1],f32>, !torch.int -> !torch.vtensor<[384,1,1],f32>
    %int-1_1166 = torch.constant.int -1
    %825 = torch.aten.unsqueeze %818, %int-1_1166 : !torch.vtensor<[384],f32>, !torch.int -> !torch.vtensor<[384,1],f32>
    %int-1_1167 = torch.constant.int -1
    %826 = torch.aten.unsqueeze %825, %int-1_1167 : !torch.vtensor<[384,1],f32>, !torch.int -> !torch.vtensor<[384,1,1],f32>
    %int1_1168 = torch.constant.int 1
    %827 = torch.aten.sub.Tensor %812, %824, %int1_1168 : !torch.vtensor<[1,384,14,14],f32>, !torch.vtensor<[384,1,1],f32>, !torch.int -> !torch.vtensor<[1,384,14,14],f32>
    %828 = torch.aten.mul.Tensor %827, %826 : !torch.vtensor<[1,384,14,14],f32>, !torch.vtensor<[384,1,1],f32> -> !torch.vtensor<[1,384,14,14],f32>
    %int-1_1169 = torch.constant.int -1
    %829 = torch.aten.unsqueeze %arg139, %int-1_1169 : !torch.vtensor<[384],f32>, !torch.int -> !torch.vtensor<[384,1],f32>
    %int-1_1170 = torch.constant.int -1
    %830 = torch.aten.unsqueeze %829, %int-1_1170 : !torch.vtensor<[384,1],f32>, !torch.int -> !torch.vtensor<[384,1,1],f32>
    %831 = torch.aten.mul.Tensor %828, %830 : !torch.vtensor<[1,384,14,14],f32>, !torch.vtensor<[384,1,1],f32> -> !torch.vtensor<[1,384,14,14],f32>
    %int-1_1171 = torch.constant.int -1
    %832 = torch.aten.unsqueeze %arg140, %int-1_1171 : !torch.vtensor<[384],f32>, !torch.int -> !torch.vtensor<[384,1],f32>
    %int-1_1172 = torch.constant.int -1
    %833 = torch.aten.unsqueeze %832, %int-1_1172 : !torch.vtensor<[384,1],f32>, !torch.int -> !torch.vtensor<[384,1,1],f32>
    %int1_1173 = torch.constant.int 1
    %834 = torch.aten.add.Tensor %831, %833, %int1_1173 : !torch.vtensor<[1,384,14,14],f32>, !torch.vtensor<[384,1,1],f32>, !torch.int -> !torch.vtensor<[1,384,14,14],f32>
    %float0.000000e00_1174 = torch.constant.float 0.000000e+00
    %float6.000000e00_1175 = torch.constant.float 6.000000e+00
    %835 = torch.aten.hardtanh %834, %float0.000000e00_1174, %float6.000000e00_1175 : !torch.vtensor<[1,384,14,14],f32>, !torch.float, !torch.float -> !torch.vtensor<[1,384,14,14],f32>
    %int1_1176 = torch.constant.int 1
    %int1_1177 = torch.constant.int 1
    %int1_1178 = torch.constant.int 1
    %int1_1179 = torch.constant.int 1
    %836 = torch.prim.ListConstruct %int1_1176, %int1_1177, %int1_1178, %int1_1179 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %float0.000000e00_1180 = torch.constant.float 0.000000e+00
    %837 = torch.aten.constant_pad_nd %835, %836, %float0.000000e00_1180 : !torch.vtensor<[1,384,14,14],f32>, !torch.list<int>, !torch.float -> !torch.vtensor<[1,384,16,16],f32>
    %none_1181 = torch.constant.none
    %int1_1182 = torch.constant.int 1
    %int1_1183 = torch.constant.int 1
    %838 = torch.prim.ListConstruct %int1_1182, %int1_1183 : (!torch.int, !torch.int) -> !torch.list<int>
    %int0_1184 = torch.constant.int 0
    %int0_1185 = torch.constant.int 0
    %839 = torch.prim.ListConstruct %int0_1184, %int0_1185 : (!torch.int, !torch.int) -> !torch.list<int>
    %int1_1186 = torch.constant.int 1
    %int1_1187 = torch.constant.int 1
    %840 = torch.prim.ListConstruct %int1_1186, %int1_1187 : (!torch.int, !torch.int) -> !torch.list<int>
    %false_1188 = torch.constant.bool false
    %int0_1189 = torch.constant.int 0
    %int0_1190 = torch.constant.int 0
    %841 = torch.prim.ListConstruct %int0_1189, %int0_1190 : (!torch.int, !torch.int) -> !torch.list<int>
    %int384_1191 = torch.constant.int 384
    %842 = torch.aten.convolution %837, %arg141, %none_1181, %838, %839, %840, %false_1188, %841, %int384_1191 : !torch.vtensor<[1,384,16,16],f32>, !torch.vtensor<[384,1,3,3],f32>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[1,384,14,14],f32>
    %int6_1192 = torch.constant.int 6
    %843 = torch.prims.convert_element_type %arg142, %int6_1192 : !torch.vtensor<[384],f32>, !torch.int -> !torch.vtensor<[384],f32>
    %int6_1193 = torch.constant.int 6
    %844 = torch.prims.convert_element_type %arg143, %int6_1193 : !torch.vtensor<[384],f32>, !torch.int -> !torch.vtensor<[384],f32>
    %float1.000000e-03_1194 = torch.constant.float 1.000000e-03
    %int1_1195 = torch.constant.int 1
    %845 = torch.aten.add.Scalar %844, %float1.000000e-03_1194, %int1_1195 : !torch.vtensor<[384],f32>, !torch.float, !torch.int -> !torch.vtensor<[384],f32>
    %846 = torch.aten.sqrt %845 : !torch.vtensor<[384],f32> -> !torch.vtensor<[384],f32>
    %847 = torch.aten.reciprocal %846 : !torch.vtensor<[384],f32> -> !torch.vtensor<[384],f32>
    %int1_1196 = torch.constant.int 1
    %848 = torch.aten.mul.Scalar %847, %int1_1196 : !torch.vtensor<[384],f32>, !torch.int -> !torch.vtensor<[384],f32>
    %int0_1197 = torch.constant.int 0
    %849 = torch.prim.ListConstruct %int0_1197 : (!torch.int) -> !torch.list<int>
    %none_1198 = torch.constant.none
    %none_1199 = torch.constant.none
    %none_1200 = torch.constant.none
    %false_1201 = torch.constant.bool false
    %850 = torch.aten.new_zeros %842, %849, %none_1198, %none_1199, %none_1200, %false_1201 : !torch.vtensor<[1,384,14,14],f32>, !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[0],f32>
    %int0_1202 = torch.constant.int 0
    %851 = torch.prim.ListConstruct %int0_1202 : (!torch.int) -> !torch.list<int>
    %none_1203 = torch.constant.none
    %none_1204 = torch.constant.none
    %none_1205 = torch.constant.none
    %false_1206 = torch.constant.bool false
    %852 = torch.aten.new_zeros %842, %851, %none_1203, %none_1204, %none_1205, %false_1206 : !torch.vtensor<[1,384,14,14],f32>, !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[0],f32>
    %int-1_1207 = torch.constant.int -1
    %853 = torch.aten.unsqueeze %843, %int-1_1207 : !torch.vtensor<[384],f32>, !torch.int -> !torch.vtensor<[384,1],f32>
    %int-1_1208 = torch.constant.int -1
    %854 = torch.aten.unsqueeze %853, %int-1_1208 : !torch.vtensor<[384,1],f32>, !torch.int -> !torch.vtensor<[384,1,1],f32>
    %int-1_1209 = torch.constant.int -1
    %855 = torch.aten.unsqueeze %848, %int-1_1209 : !torch.vtensor<[384],f32>, !torch.int -> !torch.vtensor<[384,1],f32>
    %int-1_1210 = torch.constant.int -1
    %856 = torch.aten.unsqueeze %855, %int-1_1210 : !torch.vtensor<[384,1],f32>, !torch.int -> !torch.vtensor<[384,1,1],f32>
    %int1_1211 = torch.constant.int 1
    %857 = torch.aten.sub.Tensor %842, %854, %int1_1211 : !torch.vtensor<[1,384,14,14],f32>, !torch.vtensor<[384,1,1],f32>, !torch.int -> !torch.vtensor<[1,384,14,14],f32>
    %858 = torch.aten.mul.Tensor %857, %856 : !torch.vtensor<[1,384,14,14],f32>, !torch.vtensor<[384,1,1],f32> -> !torch.vtensor<[1,384,14,14],f32>
    %int-1_1212 = torch.constant.int -1
    %859 = torch.aten.unsqueeze %arg144, %int-1_1212 : !torch.vtensor<[384],f32>, !torch.int -> !torch.vtensor<[384,1],f32>
    %int-1_1213 = torch.constant.int -1
    %860 = torch.aten.unsqueeze %859, %int-1_1213 : !torch.vtensor<[384,1],f32>, !torch.int -> !torch.vtensor<[384,1,1],f32>
    %861 = torch.aten.mul.Tensor %858, %860 : !torch.vtensor<[1,384,14,14],f32>, !torch.vtensor<[384,1,1],f32> -> !torch.vtensor<[1,384,14,14],f32>
    %int-1_1214 = torch.constant.int -1
    %862 = torch.aten.unsqueeze %arg145, %int-1_1214 : !torch.vtensor<[384],f32>, !torch.int -> !torch.vtensor<[384,1],f32>
    %int-1_1215 = torch.constant.int -1
    %863 = torch.aten.unsqueeze %862, %int-1_1215 : !torch.vtensor<[384,1],f32>, !torch.int -> !torch.vtensor<[384,1,1],f32>
    %int1_1216 = torch.constant.int 1
    %864 = torch.aten.add.Tensor %861, %863, %int1_1216 : !torch.vtensor<[1,384,14,14],f32>, !torch.vtensor<[384,1,1],f32>, !torch.int -> !torch.vtensor<[1,384,14,14],f32>
    %float0.000000e00_1217 = torch.constant.float 0.000000e+00
    %float6.000000e00_1218 = torch.constant.float 6.000000e+00
    %865 = torch.aten.hardtanh %864, %float0.000000e00_1217, %float6.000000e00_1218 : !torch.vtensor<[1,384,14,14],f32>, !torch.float, !torch.float -> !torch.vtensor<[1,384,14,14],f32>
    %int0_1219 = torch.constant.int 0
    %int0_1220 = torch.constant.int 0
    %int0_1221 = torch.constant.int 0
    %int0_1222 = torch.constant.int 0
    %866 = torch.prim.ListConstruct %int0_1219, %int0_1220, %int0_1221, %int0_1222 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %float0.000000e00_1223 = torch.constant.float 0.000000e+00
    %867 = torch.aten.constant_pad_nd %865, %866, %float0.000000e00_1223 : !torch.vtensor<[1,384,14,14],f32>, !torch.list<int>, !torch.float -> !torch.vtensor<[1,384,14,14],f32>
    %none_1224 = torch.constant.none
    %int1_1225 = torch.constant.int 1
    %int1_1226 = torch.constant.int 1
    %868 = torch.prim.ListConstruct %int1_1225, %int1_1226 : (!torch.int, !torch.int) -> !torch.list<int>
    %int0_1227 = torch.constant.int 0
    %int0_1228 = torch.constant.int 0
    %869 = torch.prim.ListConstruct %int0_1227, %int0_1228 : (!torch.int, !torch.int) -> !torch.list<int>
    %int1_1229 = torch.constant.int 1
    %int1_1230 = torch.constant.int 1
    %870 = torch.prim.ListConstruct %int1_1229, %int1_1230 : (!torch.int, !torch.int) -> !torch.list<int>
    %false_1231 = torch.constant.bool false
    %int0_1232 = torch.constant.int 0
    %int0_1233 = torch.constant.int 0
    %871 = torch.prim.ListConstruct %int0_1232, %int0_1233 : (!torch.int, !torch.int) -> !torch.list<int>
    %int1_1234 = torch.constant.int 1
    %872 = torch.aten.convolution %867, %arg146, %none_1224, %868, %869, %870, %false_1231, %871, %int1_1234 : !torch.vtensor<[1,384,14,14],f32>, !torch.vtensor<[64,384,1,1],f32>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[1,64,14,14],f32>
    %int6_1235 = torch.constant.int 6
    %873 = torch.prims.convert_element_type %arg147, %int6_1235 : !torch.vtensor<[64],f32>, !torch.int -> !torch.vtensor<[64],f32>
    %int6_1236 = torch.constant.int 6
    %874 = torch.prims.convert_element_type %arg148, %int6_1236 : !torch.vtensor<[64],f32>, !torch.int -> !torch.vtensor<[64],f32>
    %float1.000000e-03_1237 = torch.constant.float 1.000000e-03
    %int1_1238 = torch.constant.int 1
    %875 = torch.aten.add.Scalar %874, %float1.000000e-03_1237, %int1_1238 : !torch.vtensor<[64],f32>, !torch.float, !torch.int -> !torch.vtensor<[64],f32>
    %876 = torch.aten.sqrt %875 : !torch.vtensor<[64],f32> -> !torch.vtensor<[64],f32>
    %877 = torch.aten.reciprocal %876 : !torch.vtensor<[64],f32> -> !torch.vtensor<[64],f32>
    %int1_1239 = torch.constant.int 1
    %878 = torch.aten.mul.Scalar %877, %int1_1239 : !torch.vtensor<[64],f32>, !torch.int -> !torch.vtensor<[64],f32>
    %int0_1240 = torch.constant.int 0
    %879 = torch.prim.ListConstruct %int0_1240 : (!torch.int) -> !torch.list<int>
    %none_1241 = torch.constant.none
    %none_1242 = torch.constant.none
    %none_1243 = torch.constant.none
    %false_1244 = torch.constant.bool false
    %880 = torch.aten.new_zeros %872, %879, %none_1241, %none_1242, %none_1243, %false_1244 : !torch.vtensor<[1,64,14,14],f32>, !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[0],f32>
    %int0_1245 = torch.constant.int 0
    %881 = torch.prim.ListConstruct %int0_1245 : (!torch.int) -> !torch.list<int>
    %none_1246 = torch.constant.none
    %none_1247 = torch.constant.none
    %none_1248 = torch.constant.none
    %false_1249 = torch.constant.bool false
    %882 = torch.aten.new_zeros %872, %881, %none_1246, %none_1247, %none_1248, %false_1249 : !torch.vtensor<[1,64,14,14],f32>, !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[0],f32>
    %int-1_1250 = torch.constant.int -1
    %883 = torch.aten.unsqueeze %873, %int-1_1250 : !torch.vtensor<[64],f32>, !torch.int -> !torch.vtensor<[64,1],f32>
    %int-1_1251 = torch.constant.int -1
    %884 = torch.aten.unsqueeze %883, %int-1_1251 : !torch.vtensor<[64,1],f32>, !torch.int -> !torch.vtensor<[64,1,1],f32>
    %int-1_1252 = torch.constant.int -1
    %885 = torch.aten.unsqueeze %878, %int-1_1252 : !torch.vtensor<[64],f32>, !torch.int -> !torch.vtensor<[64,1],f32>
    %int-1_1253 = torch.constant.int -1
    %886 = torch.aten.unsqueeze %885, %int-1_1253 : !torch.vtensor<[64,1],f32>, !torch.int -> !torch.vtensor<[64,1,1],f32>
    %int1_1254 = torch.constant.int 1
    %887 = torch.aten.sub.Tensor %872, %884, %int1_1254 : !torch.vtensor<[1,64,14,14],f32>, !torch.vtensor<[64,1,1],f32>, !torch.int -> !torch.vtensor<[1,64,14,14],f32>
    %888 = torch.aten.mul.Tensor %887, %886 : !torch.vtensor<[1,64,14,14],f32>, !torch.vtensor<[64,1,1],f32> -> !torch.vtensor<[1,64,14,14],f32>
    %int-1_1255 = torch.constant.int -1
    %889 = torch.aten.unsqueeze %arg149, %int-1_1255 : !torch.vtensor<[64],f32>, !torch.int -> !torch.vtensor<[64,1],f32>
    %int-1_1256 = torch.constant.int -1
    %890 = torch.aten.unsqueeze %889, %int-1_1256 : !torch.vtensor<[64,1],f32>, !torch.int -> !torch.vtensor<[64,1,1],f32>
    %891 = torch.aten.mul.Tensor %888, %890 : !torch.vtensor<[1,64,14,14],f32>, !torch.vtensor<[64,1,1],f32> -> !torch.vtensor<[1,64,14,14],f32>
    %int-1_1257 = torch.constant.int -1
    %892 = torch.aten.unsqueeze %arg150, %int-1_1257 : !torch.vtensor<[64],f32>, !torch.int -> !torch.vtensor<[64,1],f32>
    %int-1_1258 = torch.constant.int -1
    %893 = torch.aten.unsqueeze %892, %int-1_1258 : !torch.vtensor<[64,1],f32>, !torch.int -> !torch.vtensor<[64,1,1],f32>
    %int1_1259 = torch.constant.int 1
    %894 = torch.aten.add.Tensor %891, %893, %int1_1259 : !torch.vtensor<[1,64,14,14],f32>, !torch.vtensor<[64,1,1],f32>, !torch.int -> !torch.vtensor<[1,64,14,14],f32>
    %int1_1260 = torch.constant.int 1
    %895 = torch.aten.add.Tensor %805, %894, %int1_1260 : !torch.vtensor<[1,64,14,14],f32>, !torch.vtensor<[1,64,14,14],f32>, !torch.int -> !torch.vtensor<[1,64,14,14],f32>
    %int0_1261 = torch.constant.int 0
    %int0_1262 = torch.constant.int 0
    %int0_1263 = torch.constant.int 0
    %int0_1264 = torch.constant.int 0
    %896 = torch.prim.ListConstruct %int0_1261, %int0_1262, %int0_1263, %int0_1264 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %float0.000000e00_1265 = torch.constant.float 0.000000e+00
    %897 = torch.aten.constant_pad_nd %895, %896, %float0.000000e00_1265 : !torch.vtensor<[1,64,14,14],f32>, !torch.list<int>, !torch.float -> !torch.vtensor<[1,64,14,14],f32>
    %none_1266 = torch.constant.none
    %int1_1267 = torch.constant.int 1
    %int1_1268 = torch.constant.int 1
    %898 = torch.prim.ListConstruct %int1_1267, %int1_1268 : (!torch.int, !torch.int) -> !torch.list<int>
    %int0_1269 = torch.constant.int 0
    %int0_1270 = torch.constant.int 0
    %899 = torch.prim.ListConstruct %int0_1269, %int0_1270 : (!torch.int, !torch.int) -> !torch.list<int>
    %int1_1271 = torch.constant.int 1
    %int1_1272 = torch.constant.int 1
    %900 = torch.prim.ListConstruct %int1_1271, %int1_1272 : (!torch.int, !torch.int) -> !torch.list<int>
    %false_1273 = torch.constant.bool false
    %int0_1274 = torch.constant.int 0
    %int0_1275 = torch.constant.int 0
    %901 = torch.prim.ListConstruct %int0_1274, %int0_1275 : (!torch.int, !torch.int) -> !torch.list<int>
    %int1_1276 = torch.constant.int 1
    %902 = torch.aten.convolution %897, %arg151, %none_1266, %898, %899, %900, %false_1273, %901, %int1_1276 : !torch.vtensor<[1,64,14,14],f32>, !torch.vtensor<[384,64,1,1],f32>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[1,384,14,14],f32>
    %int6_1277 = torch.constant.int 6
    %903 = torch.prims.convert_element_type %arg152, %int6_1277 : !torch.vtensor<[384],f32>, !torch.int -> !torch.vtensor<[384],f32>
    %int6_1278 = torch.constant.int 6
    %904 = torch.prims.convert_element_type %arg153, %int6_1278 : !torch.vtensor<[384],f32>, !torch.int -> !torch.vtensor<[384],f32>
    %float1.000000e-03_1279 = torch.constant.float 1.000000e-03
    %int1_1280 = torch.constant.int 1
    %905 = torch.aten.add.Scalar %904, %float1.000000e-03_1279, %int1_1280 : !torch.vtensor<[384],f32>, !torch.float, !torch.int -> !torch.vtensor<[384],f32>
    %906 = torch.aten.sqrt %905 : !torch.vtensor<[384],f32> -> !torch.vtensor<[384],f32>
    %907 = torch.aten.reciprocal %906 : !torch.vtensor<[384],f32> -> !torch.vtensor<[384],f32>
    %int1_1281 = torch.constant.int 1
    %908 = torch.aten.mul.Scalar %907, %int1_1281 : !torch.vtensor<[384],f32>, !torch.int -> !torch.vtensor<[384],f32>
    %int0_1282 = torch.constant.int 0
    %909 = torch.prim.ListConstruct %int0_1282 : (!torch.int) -> !torch.list<int>
    %none_1283 = torch.constant.none
    %none_1284 = torch.constant.none
    %none_1285 = torch.constant.none
    %false_1286 = torch.constant.bool false
    %910 = torch.aten.new_zeros %902, %909, %none_1283, %none_1284, %none_1285, %false_1286 : !torch.vtensor<[1,384,14,14],f32>, !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[0],f32>
    %int0_1287 = torch.constant.int 0
    %911 = torch.prim.ListConstruct %int0_1287 : (!torch.int) -> !torch.list<int>
    %none_1288 = torch.constant.none
    %none_1289 = torch.constant.none
    %none_1290 = torch.constant.none
    %false_1291 = torch.constant.bool false
    %912 = torch.aten.new_zeros %902, %911, %none_1288, %none_1289, %none_1290, %false_1291 : !torch.vtensor<[1,384,14,14],f32>, !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[0],f32>
    %int-1_1292 = torch.constant.int -1
    %913 = torch.aten.unsqueeze %903, %int-1_1292 : !torch.vtensor<[384],f32>, !torch.int -> !torch.vtensor<[384,1],f32>
    %int-1_1293 = torch.constant.int -1
    %914 = torch.aten.unsqueeze %913, %int-1_1293 : !torch.vtensor<[384,1],f32>, !torch.int -> !torch.vtensor<[384,1,1],f32>
    %int-1_1294 = torch.constant.int -1
    %915 = torch.aten.unsqueeze %908, %int-1_1294 : !torch.vtensor<[384],f32>, !torch.int -> !torch.vtensor<[384,1],f32>
    %int-1_1295 = torch.constant.int -1
    %916 = torch.aten.unsqueeze %915, %int-1_1295 : !torch.vtensor<[384,1],f32>, !torch.int -> !torch.vtensor<[384,1,1],f32>
    %int1_1296 = torch.constant.int 1
    %917 = torch.aten.sub.Tensor %902, %914, %int1_1296 : !torch.vtensor<[1,384,14,14],f32>, !torch.vtensor<[384,1,1],f32>, !torch.int -> !torch.vtensor<[1,384,14,14],f32>
    %918 = torch.aten.mul.Tensor %917, %916 : !torch.vtensor<[1,384,14,14],f32>, !torch.vtensor<[384,1,1],f32> -> !torch.vtensor<[1,384,14,14],f32>
    %int-1_1297 = torch.constant.int -1
    %919 = torch.aten.unsqueeze %arg154, %int-1_1297 : !torch.vtensor<[384],f32>, !torch.int -> !torch.vtensor<[384,1],f32>
    %int-1_1298 = torch.constant.int -1
    %920 = torch.aten.unsqueeze %919, %int-1_1298 : !torch.vtensor<[384,1],f32>, !torch.int -> !torch.vtensor<[384,1,1],f32>
    %921 = torch.aten.mul.Tensor %918, %920 : !torch.vtensor<[1,384,14,14],f32>, !torch.vtensor<[384,1,1],f32> -> !torch.vtensor<[1,384,14,14],f32>
    %int-1_1299 = torch.constant.int -1
    %922 = torch.aten.unsqueeze %arg155, %int-1_1299 : !torch.vtensor<[384],f32>, !torch.int -> !torch.vtensor<[384,1],f32>
    %int-1_1300 = torch.constant.int -1
    %923 = torch.aten.unsqueeze %922, %int-1_1300 : !torch.vtensor<[384,1],f32>, !torch.int -> !torch.vtensor<[384,1,1],f32>
    %int1_1301 = torch.constant.int 1
    %924 = torch.aten.add.Tensor %921, %923, %int1_1301 : !torch.vtensor<[1,384,14,14],f32>, !torch.vtensor<[384,1,1],f32>, !torch.int -> !torch.vtensor<[1,384,14,14],f32>
    %float0.000000e00_1302 = torch.constant.float 0.000000e+00
    %float6.000000e00_1303 = torch.constant.float 6.000000e+00
    %925 = torch.aten.hardtanh %924, %float0.000000e00_1302, %float6.000000e00_1303 : !torch.vtensor<[1,384,14,14],f32>, !torch.float, !torch.float -> !torch.vtensor<[1,384,14,14],f32>
    %int1_1304 = torch.constant.int 1
    %int1_1305 = torch.constant.int 1
    %int1_1306 = torch.constant.int 1
    %int1_1307 = torch.constant.int 1
    %926 = torch.prim.ListConstruct %int1_1304, %int1_1305, %int1_1306, %int1_1307 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %float0.000000e00_1308 = torch.constant.float 0.000000e+00
    %927 = torch.aten.constant_pad_nd %925, %926, %float0.000000e00_1308 : !torch.vtensor<[1,384,14,14],f32>, !torch.list<int>, !torch.float -> !torch.vtensor<[1,384,16,16],f32>
    %none_1309 = torch.constant.none
    %int1_1310 = torch.constant.int 1
    %int1_1311 = torch.constant.int 1
    %928 = torch.prim.ListConstruct %int1_1310, %int1_1311 : (!torch.int, !torch.int) -> !torch.list<int>
    %int0_1312 = torch.constant.int 0
    %int0_1313 = torch.constant.int 0
    %929 = torch.prim.ListConstruct %int0_1312, %int0_1313 : (!torch.int, !torch.int) -> !torch.list<int>
    %int1_1314 = torch.constant.int 1
    %int1_1315 = torch.constant.int 1
    %930 = torch.prim.ListConstruct %int1_1314, %int1_1315 : (!torch.int, !torch.int) -> !torch.list<int>
    %false_1316 = torch.constant.bool false
    %int0_1317 = torch.constant.int 0
    %int0_1318 = torch.constant.int 0
    %931 = torch.prim.ListConstruct %int0_1317, %int0_1318 : (!torch.int, !torch.int) -> !torch.list<int>
    %int384_1319 = torch.constant.int 384
    %932 = torch.aten.convolution %927, %arg156, %none_1309, %928, %929, %930, %false_1316, %931, %int384_1319 : !torch.vtensor<[1,384,16,16],f32>, !torch.vtensor<[384,1,3,3],f32>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[1,384,14,14],f32>
    %int6_1320 = torch.constant.int 6
    %933 = torch.prims.convert_element_type %arg157, %int6_1320 : !torch.vtensor<[384],f32>, !torch.int -> !torch.vtensor<[384],f32>
    %int6_1321 = torch.constant.int 6
    %934 = torch.prims.convert_element_type %arg158, %int6_1321 : !torch.vtensor<[384],f32>, !torch.int -> !torch.vtensor<[384],f32>
    %float1.000000e-03_1322 = torch.constant.float 1.000000e-03
    %int1_1323 = torch.constant.int 1
    %935 = torch.aten.add.Scalar %934, %float1.000000e-03_1322, %int1_1323 : !torch.vtensor<[384],f32>, !torch.float, !torch.int -> !torch.vtensor<[384],f32>
    %936 = torch.aten.sqrt %935 : !torch.vtensor<[384],f32> -> !torch.vtensor<[384],f32>
    %937 = torch.aten.reciprocal %936 : !torch.vtensor<[384],f32> -> !torch.vtensor<[384],f32>
    %int1_1324 = torch.constant.int 1
    %938 = torch.aten.mul.Scalar %937, %int1_1324 : !torch.vtensor<[384],f32>, !torch.int -> !torch.vtensor<[384],f32>
    %int0_1325 = torch.constant.int 0
    %939 = torch.prim.ListConstruct %int0_1325 : (!torch.int) -> !torch.list<int>
    %none_1326 = torch.constant.none
    %none_1327 = torch.constant.none
    %none_1328 = torch.constant.none
    %false_1329 = torch.constant.bool false
    %940 = torch.aten.new_zeros %932, %939, %none_1326, %none_1327, %none_1328, %false_1329 : !torch.vtensor<[1,384,14,14],f32>, !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[0],f32>
    %int0_1330 = torch.constant.int 0
    %941 = torch.prim.ListConstruct %int0_1330 : (!torch.int) -> !torch.list<int>
    %none_1331 = torch.constant.none
    %none_1332 = torch.constant.none
    %none_1333 = torch.constant.none
    %false_1334 = torch.constant.bool false
    %942 = torch.aten.new_zeros %932, %941, %none_1331, %none_1332, %none_1333, %false_1334 : !torch.vtensor<[1,384,14,14],f32>, !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[0],f32>
    %int-1_1335 = torch.constant.int -1
    %943 = torch.aten.unsqueeze %933, %int-1_1335 : !torch.vtensor<[384],f32>, !torch.int -> !torch.vtensor<[384,1],f32>
    %int-1_1336 = torch.constant.int -1
    %944 = torch.aten.unsqueeze %943, %int-1_1336 : !torch.vtensor<[384,1],f32>, !torch.int -> !torch.vtensor<[384,1,1],f32>
    %int-1_1337 = torch.constant.int -1
    %945 = torch.aten.unsqueeze %938, %int-1_1337 : !torch.vtensor<[384],f32>, !torch.int -> !torch.vtensor<[384,1],f32>
    %int-1_1338 = torch.constant.int -1
    %946 = torch.aten.unsqueeze %945, %int-1_1338 : !torch.vtensor<[384,1],f32>, !torch.int -> !torch.vtensor<[384,1,1],f32>
    %int1_1339 = torch.constant.int 1
    %947 = torch.aten.sub.Tensor %932, %944, %int1_1339 : !torch.vtensor<[1,384,14,14],f32>, !torch.vtensor<[384,1,1],f32>, !torch.int -> !torch.vtensor<[1,384,14,14],f32>
    %948 = torch.aten.mul.Tensor %947, %946 : !torch.vtensor<[1,384,14,14],f32>, !torch.vtensor<[384,1,1],f32> -> !torch.vtensor<[1,384,14,14],f32>
    %int-1_1340 = torch.constant.int -1
    %949 = torch.aten.unsqueeze %arg159, %int-1_1340 : !torch.vtensor<[384],f32>, !torch.int -> !torch.vtensor<[384,1],f32>
    %int-1_1341 = torch.constant.int -1
    %950 = torch.aten.unsqueeze %949, %int-1_1341 : !torch.vtensor<[384,1],f32>, !torch.int -> !torch.vtensor<[384,1,1],f32>
    %951 = torch.aten.mul.Tensor %948, %950 : !torch.vtensor<[1,384,14,14],f32>, !torch.vtensor<[384,1,1],f32> -> !torch.vtensor<[1,384,14,14],f32>
    %int-1_1342 = torch.constant.int -1
    %952 = torch.aten.unsqueeze %arg160, %int-1_1342 : !torch.vtensor<[384],f32>, !torch.int -> !torch.vtensor<[384,1],f32>
    %int-1_1343 = torch.constant.int -1
    %953 = torch.aten.unsqueeze %952, %int-1_1343 : !torch.vtensor<[384,1],f32>, !torch.int -> !torch.vtensor<[384,1,1],f32>
    %int1_1344 = torch.constant.int 1
    %954 = torch.aten.add.Tensor %951, %953, %int1_1344 : !torch.vtensor<[1,384,14,14],f32>, !torch.vtensor<[384,1,1],f32>, !torch.int -> !torch.vtensor<[1,384,14,14],f32>
    %float0.000000e00_1345 = torch.constant.float 0.000000e+00
    %float6.000000e00_1346 = torch.constant.float 6.000000e+00
    %955 = torch.aten.hardtanh %954, %float0.000000e00_1345, %float6.000000e00_1346 : !torch.vtensor<[1,384,14,14],f32>, !torch.float, !torch.float -> !torch.vtensor<[1,384,14,14],f32>
    %int0_1347 = torch.constant.int 0
    %int0_1348 = torch.constant.int 0
    %int0_1349 = torch.constant.int 0
    %int0_1350 = torch.constant.int 0
    %956 = torch.prim.ListConstruct %int0_1347, %int0_1348, %int0_1349, %int0_1350 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %float0.000000e00_1351 = torch.constant.float 0.000000e+00
    %957 = torch.aten.constant_pad_nd %955, %956, %float0.000000e00_1351 : !torch.vtensor<[1,384,14,14],f32>, !torch.list<int>, !torch.float -> !torch.vtensor<[1,384,14,14],f32>
    %none_1352 = torch.constant.none
    %int1_1353 = torch.constant.int 1
    %int1_1354 = torch.constant.int 1
    %958 = torch.prim.ListConstruct %int1_1353, %int1_1354 : (!torch.int, !torch.int) -> !torch.list<int>
    %int0_1355 = torch.constant.int 0
    %int0_1356 = torch.constant.int 0
    %959 = torch.prim.ListConstruct %int0_1355, %int0_1356 : (!torch.int, !torch.int) -> !torch.list<int>
    %int1_1357 = torch.constant.int 1
    %int1_1358 = torch.constant.int 1
    %960 = torch.prim.ListConstruct %int1_1357, %int1_1358 : (!torch.int, !torch.int) -> !torch.list<int>
    %false_1359 = torch.constant.bool false
    %int0_1360 = torch.constant.int 0
    %int0_1361 = torch.constant.int 0
    %961 = torch.prim.ListConstruct %int0_1360, %int0_1361 : (!torch.int, !torch.int) -> !torch.list<int>
    %int1_1362 = torch.constant.int 1
    %962 = torch.aten.convolution %957, %arg161, %none_1352, %958, %959, %960, %false_1359, %961, %int1_1362 : !torch.vtensor<[1,384,14,14],f32>, !torch.vtensor<[96,384,1,1],f32>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[1,96,14,14],f32>
    %int6_1363 = torch.constant.int 6
    %963 = torch.prims.convert_element_type %arg162, %int6_1363 : !torch.vtensor<[96],f32>, !torch.int -> !torch.vtensor<[96],f32>
    %int6_1364 = torch.constant.int 6
    %964 = torch.prims.convert_element_type %arg163, %int6_1364 : !torch.vtensor<[96],f32>, !torch.int -> !torch.vtensor<[96],f32>
    %float1.000000e-03_1365 = torch.constant.float 1.000000e-03
    %int1_1366 = torch.constant.int 1
    %965 = torch.aten.add.Scalar %964, %float1.000000e-03_1365, %int1_1366 : !torch.vtensor<[96],f32>, !torch.float, !torch.int -> !torch.vtensor<[96],f32>
    %966 = torch.aten.sqrt %965 : !torch.vtensor<[96],f32> -> !torch.vtensor<[96],f32>
    %967 = torch.aten.reciprocal %966 : !torch.vtensor<[96],f32> -> !torch.vtensor<[96],f32>
    %int1_1367 = torch.constant.int 1
    %968 = torch.aten.mul.Scalar %967, %int1_1367 : !torch.vtensor<[96],f32>, !torch.int -> !torch.vtensor<[96],f32>
    %int0_1368 = torch.constant.int 0
    %969 = torch.prim.ListConstruct %int0_1368 : (!torch.int) -> !torch.list<int>
    %none_1369 = torch.constant.none
    %none_1370 = torch.constant.none
    %none_1371 = torch.constant.none
    %false_1372 = torch.constant.bool false
    %970 = torch.aten.new_zeros %962, %969, %none_1369, %none_1370, %none_1371, %false_1372 : !torch.vtensor<[1,96,14,14],f32>, !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[0],f32>
    %int0_1373 = torch.constant.int 0
    %971 = torch.prim.ListConstruct %int0_1373 : (!torch.int) -> !torch.list<int>
    %none_1374 = torch.constant.none
    %none_1375 = torch.constant.none
    %none_1376 = torch.constant.none
    %false_1377 = torch.constant.bool false
    %972 = torch.aten.new_zeros %962, %971, %none_1374, %none_1375, %none_1376, %false_1377 : !torch.vtensor<[1,96,14,14],f32>, !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[0],f32>
    %int-1_1378 = torch.constant.int -1
    %973 = torch.aten.unsqueeze %963, %int-1_1378 : !torch.vtensor<[96],f32>, !torch.int -> !torch.vtensor<[96,1],f32>
    %int-1_1379 = torch.constant.int -1
    %974 = torch.aten.unsqueeze %973, %int-1_1379 : !torch.vtensor<[96,1],f32>, !torch.int -> !torch.vtensor<[96,1,1],f32>
    %int-1_1380 = torch.constant.int -1
    %975 = torch.aten.unsqueeze %968, %int-1_1380 : !torch.vtensor<[96],f32>, !torch.int -> !torch.vtensor<[96,1],f32>
    %int-1_1381 = torch.constant.int -1
    %976 = torch.aten.unsqueeze %975, %int-1_1381 : !torch.vtensor<[96,1],f32>, !torch.int -> !torch.vtensor<[96,1,1],f32>
    %int1_1382 = torch.constant.int 1
    %977 = torch.aten.sub.Tensor %962, %974, %int1_1382 : !torch.vtensor<[1,96,14,14],f32>, !torch.vtensor<[96,1,1],f32>, !torch.int -> !torch.vtensor<[1,96,14,14],f32>
    %978 = torch.aten.mul.Tensor %977, %976 : !torch.vtensor<[1,96,14,14],f32>, !torch.vtensor<[96,1,1],f32> -> !torch.vtensor<[1,96,14,14],f32>
    %int-1_1383 = torch.constant.int -1
    %979 = torch.aten.unsqueeze %arg164, %int-1_1383 : !torch.vtensor<[96],f32>, !torch.int -> !torch.vtensor<[96,1],f32>
    %int-1_1384 = torch.constant.int -1
    %980 = torch.aten.unsqueeze %979, %int-1_1384 : !torch.vtensor<[96,1],f32>, !torch.int -> !torch.vtensor<[96,1,1],f32>
    %981 = torch.aten.mul.Tensor %978, %980 : !torch.vtensor<[1,96,14,14],f32>, !torch.vtensor<[96,1,1],f32> -> !torch.vtensor<[1,96,14,14],f32>
    %int-1_1385 = torch.constant.int -1
    %982 = torch.aten.unsqueeze %arg165, %int-1_1385 : !torch.vtensor<[96],f32>, !torch.int -> !torch.vtensor<[96,1],f32>
    %int-1_1386 = torch.constant.int -1
    %983 = torch.aten.unsqueeze %982, %int-1_1386 : !torch.vtensor<[96,1],f32>, !torch.int -> !torch.vtensor<[96,1,1],f32>
    %int1_1387 = torch.constant.int 1
    %984 = torch.aten.add.Tensor %981, %983, %int1_1387 : !torch.vtensor<[1,96,14,14],f32>, !torch.vtensor<[96,1,1],f32>, !torch.int -> !torch.vtensor<[1,96,14,14],f32>
    %int0_1388 = torch.constant.int 0
    %int0_1389 = torch.constant.int 0
    %int0_1390 = torch.constant.int 0
    %int0_1391 = torch.constant.int 0
    %985 = torch.prim.ListConstruct %int0_1388, %int0_1389, %int0_1390, %int0_1391 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %float0.000000e00_1392 = torch.constant.float 0.000000e+00
    %986 = torch.aten.constant_pad_nd %984, %985, %float0.000000e00_1392 : !torch.vtensor<[1,96,14,14],f32>, !torch.list<int>, !torch.float -> !torch.vtensor<[1,96,14,14],f32>
    %none_1393 = torch.constant.none
    %int1_1394 = torch.constant.int 1
    %int1_1395 = torch.constant.int 1
    %987 = torch.prim.ListConstruct %int1_1394, %int1_1395 : (!torch.int, !torch.int) -> !torch.list<int>
    %int0_1396 = torch.constant.int 0
    %int0_1397 = torch.constant.int 0
    %988 = torch.prim.ListConstruct %int0_1396, %int0_1397 : (!torch.int, !torch.int) -> !torch.list<int>
    %int1_1398 = torch.constant.int 1
    %int1_1399 = torch.constant.int 1
    %989 = torch.prim.ListConstruct %int1_1398, %int1_1399 : (!torch.int, !torch.int) -> !torch.list<int>
    %false_1400 = torch.constant.bool false
    %int0_1401 = torch.constant.int 0
    %int0_1402 = torch.constant.int 0
    %990 = torch.prim.ListConstruct %int0_1401, %int0_1402 : (!torch.int, !torch.int) -> !torch.list<int>
    %int1_1403 = torch.constant.int 1
    %991 = torch.aten.convolution %986, %arg166, %none_1393, %987, %988, %989, %false_1400, %990, %int1_1403 : !torch.vtensor<[1,96,14,14],f32>, !torch.vtensor<[576,96,1,1],f32>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[1,576,14,14],f32>
    %int6_1404 = torch.constant.int 6
    %992 = torch.prims.convert_element_type %arg167, %int6_1404 : !torch.vtensor<[576],f32>, !torch.int -> !torch.vtensor<[576],f32>
    %int6_1405 = torch.constant.int 6
    %993 = torch.prims.convert_element_type %arg168, %int6_1405 : !torch.vtensor<[576],f32>, !torch.int -> !torch.vtensor<[576],f32>
    %float1.000000e-03_1406 = torch.constant.float 1.000000e-03
    %int1_1407 = torch.constant.int 1
    %994 = torch.aten.add.Scalar %993, %float1.000000e-03_1406, %int1_1407 : !torch.vtensor<[576],f32>, !torch.float, !torch.int -> !torch.vtensor<[576],f32>
    %995 = torch.aten.sqrt %994 : !torch.vtensor<[576],f32> -> !torch.vtensor<[576],f32>
    %996 = torch.aten.reciprocal %995 : !torch.vtensor<[576],f32> -> !torch.vtensor<[576],f32>
    %int1_1408 = torch.constant.int 1
    %997 = torch.aten.mul.Scalar %996, %int1_1408 : !torch.vtensor<[576],f32>, !torch.int -> !torch.vtensor<[576],f32>
    %int0_1409 = torch.constant.int 0
    %998 = torch.prim.ListConstruct %int0_1409 : (!torch.int) -> !torch.list<int>
    %none_1410 = torch.constant.none
    %none_1411 = torch.constant.none
    %none_1412 = torch.constant.none
    %false_1413 = torch.constant.bool false
    %999 = torch.aten.new_zeros %991, %998, %none_1410, %none_1411, %none_1412, %false_1413 : !torch.vtensor<[1,576,14,14],f32>, !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[0],f32>
    %int0_1414 = torch.constant.int 0
    %1000 = torch.prim.ListConstruct %int0_1414 : (!torch.int) -> !torch.list<int>
    %none_1415 = torch.constant.none
    %none_1416 = torch.constant.none
    %none_1417 = torch.constant.none
    %false_1418 = torch.constant.bool false
    %1001 = torch.aten.new_zeros %991, %1000, %none_1415, %none_1416, %none_1417, %false_1418 : !torch.vtensor<[1,576,14,14],f32>, !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[0],f32>
    %int-1_1419 = torch.constant.int -1
    %1002 = torch.aten.unsqueeze %992, %int-1_1419 : !torch.vtensor<[576],f32>, !torch.int -> !torch.vtensor<[576,1],f32>
    %int-1_1420 = torch.constant.int -1
    %1003 = torch.aten.unsqueeze %1002, %int-1_1420 : !torch.vtensor<[576,1],f32>, !torch.int -> !torch.vtensor<[576,1,1],f32>
    %int-1_1421 = torch.constant.int -1
    %1004 = torch.aten.unsqueeze %997, %int-1_1421 : !torch.vtensor<[576],f32>, !torch.int -> !torch.vtensor<[576,1],f32>
    %int-1_1422 = torch.constant.int -1
    %1005 = torch.aten.unsqueeze %1004, %int-1_1422 : !torch.vtensor<[576,1],f32>, !torch.int -> !torch.vtensor<[576,1,1],f32>
    %int1_1423 = torch.constant.int 1
    %1006 = torch.aten.sub.Tensor %991, %1003, %int1_1423 : !torch.vtensor<[1,576,14,14],f32>, !torch.vtensor<[576,1,1],f32>, !torch.int -> !torch.vtensor<[1,576,14,14],f32>
    %1007 = torch.aten.mul.Tensor %1006, %1005 : !torch.vtensor<[1,576,14,14],f32>, !torch.vtensor<[576,1,1],f32> -> !torch.vtensor<[1,576,14,14],f32>
    %int-1_1424 = torch.constant.int -1
    %1008 = torch.aten.unsqueeze %arg169, %int-1_1424 : !torch.vtensor<[576],f32>, !torch.int -> !torch.vtensor<[576,1],f32>
    %int-1_1425 = torch.constant.int -1
    %1009 = torch.aten.unsqueeze %1008, %int-1_1425 : !torch.vtensor<[576,1],f32>, !torch.int -> !torch.vtensor<[576,1,1],f32>
    %1010 = torch.aten.mul.Tensor %1007, %1009 : !torch.vtensor<[1,576,14,14],f32>, !torch.vtensor<[576,1,1],f32> -> !torch.vtensor<[1,576,14,14],f32>
    %int-1_1426 = torch.constant.int -1
    %1011 = torch.aten.unsqueeze %arg170, %int-1_1426 : !torch.vtensor<[576],f32>, !torch.int -> !torch.vtensor<[576,1],f32>
    %int-1_1427 = torch.constant.int -1
    %1012 = torch.aten.unsqueeze %1011, %int-1_1427 : !torch.vtensor<[576,1],f32>, !torch.int -> !torch.vtensor<[576,1,1],f32>
    %int1_1428 = torch.constant.int 1
    %1013 = torch.aten.add.Tensor %1010, %1012, %int1_1428 : !torch.vtensor<[1,576,14,14],f32>, !torch.vtensor<[576,1,1],f32>, !torch.int -> !torch.vtensor<[1,576,14,14],f32>
    %float0.000000e00_1429 = torch.constant.float 0.000000e+00
    %float6.000000e00_1430 = torch.constant.float 6.000000e+00
    %1014 = torch.aten.hardtanh %1013, %float0.000000e00_1429, %float6.000000e00_1430 : !torch.vtensor<[1,576,14,14],f32>, !torch.float, !torch.float -> !torch.vtensor<[1,576,14,14],f32>
    %int1_1431 = torch.constant.int 1
    %int1_1432 = torch.constant.int 1
    %int1_1433 = torch.constant.int 1
    %int1_1434 = torch.constant.int 1
    %1015 = torch.prim.ListConstruct %int1_1431, %int1_1432, %int1_1433, %int1_1434 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %float0.000000e00_1435 = torch.constant.float 0.000000e+00
    %1016 = torch.aten.constant_pad_nd %1014, %1015, %float0.000000e00_1435 : !torch.vtensor<[1,576,14,14],f32>, !torch.list<int>, !torch.float -> !torch.vtensor<[1,576,16,16],f32>
    %none_1436 = torch.constant.none
    %int1_1437 = torch.constant.int 1
    %int1_1438 = torch.constant.int 1
    %1017 = torch.prim.ListConstruct %int1_1437, %int1_1438 : (!torch.int, !torch.int) -> !torch.list<int>
    %int0_1439 = torch.constant.int 0
    %int0_1440 = torch.constant.int 0
    %1018 = torch.prim.ListConstruct %int0_1439, %int0_1440 : (!torch.int, !torch.int) -> !torch.list<int>
    %int1_1441 = torch.constant.int 1
    %int1_1442 = torch.constant.int 1
    %1019 = torch.prim.ListConstruct %int1_1441, %int1_1442 : (!torch.int, !torch.int) -> !torch.list<int>
    %false_1443 = torch.constant.bool false
    %int0_1444 = torch.constant.int 0
    %int0_1445 = torch.constant.int 0
    %1020 = torch.prim.ListConstruct %int0_1444, %int0_1445 : (!torch.int, !torch.int) -> !torch.list<int>
    %int576 = torch.constant.int 576
    %1021 = torch.aten.convolution %1016, %arg171, %none_1436, %1017, %1018, %1019, %false_1443, %1020, %int576 : !torch.vtensor<[1,576,16,16],f32>, !torch.vtensor<[576,1,3,3],f32>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[1,576,14,14],f32>
    %int6_1446 = torch.constant.int 6
    %1022 = torch.prims.convert_element_type %arg172, %int6_1446 : !torch.vtensor<[576],f32>, !torch.int -> !torch.vtensor<[576],f32>
    %int6_1447 = torch.constant.int 6
    %1023 = torch.prims.convert_element_type %arg173, %int6_1447 : !torch.vtensor<[576],f32>, !torch.int -> !torch.vtensor<[576],f32>
    %float1.000000e-03_1448 = torch.constant.float 1.000000e-03
    %int1_1449 = torch.constant.int 1
    %1024 = torch.aten.add.Scalar %1023, %float1.000000e-03_1448, %int1_1449 : !torch.vtensor<[576],f32>, !torch.float, !torch.int -> !torch.vtensor<[576],f32>
    %1025 = torch.aten.sqrt %1024 : !torch.vtensor<[576],f32> -> !torch.vtensor<[576],f32>
    %1026 = torch.aten.reciprocal %1025 : !torch.vtensor<[576],f32> -> !torch.vtensor<[576],f32>
    %int1_1450 = torch.constant.int 1
    %1027 = torch.aten.mul.Scalar %1026, %int1_1450 : !torch.vtensor<[576],f32>, !torch.int -> !torch.vtensor<[576],f32>
    %int0_1451 = torch.constant.int 0
    %1028 = torch.prim.ListConstruct %int0_1451 : (!torch.int) -> !torch.list<int>
    %none_1452 = torch.constant.none
    %none_1453 = torch.constant.none
    %none_1454 = torch.constant.none
    %false_1455 = torch.constant.bool false
    %1029 = torch.aten.new_zeros %1021, %1028, %none_1452, %none_1453, %none_1454, %false_1455 : !torch.vtensor<[1,576,14,14],f32>, !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[0],f32>
    %int0_1456 = torch.constant.int 0
    %1030 = torch.prim.ListConstruct %int0_1456 : (!torch.int) -> !torch.list<int>
    %none_1457 = torch.constant.none
    %none_1458 = torch.constant.none
    %none_1459 = torch.constant.none
    %false_1460 = torch.constant.bool false
    %1031 = torch.aten.new_zeros %1021, %1030, %none_1457, %none_1458, %none_1459, %false_1460 : !torch.vtensor<[1,576,14,14],f32>, !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[0],f32>
    %int-1_1461 = torch.constant.int -1
    %1032 = torch.aten.unsqueeze %1022, %int-1_1461 : !torch.vtensor<[576],f32>, !torch.int -> !torch.vtensor<[576,1],f32>
    %int-1_1462 = torch.constant.int -1
    %1033 = torch.aten.unsqueeze %1032, %int-1_1462 : !torch.vtensor<[576,1],f32>, !torch.int -> !torch.vtensor<[576,1,1],f32>
    %int-1_1463 = torch.constant.int -1
    %1034 = torch.aten.unsqueeze %1027, %int-1_1463 : !torch.vtensor<[576],f32>, !torch.int -> !torch.vtensor<[576,1],f32>
    %int-1_1464 = torch.constant.int -1
    %1035 = torch.aten.unsqueeze %1034, %int-1_1464 : !torch.vtensor<[576,1],f32>, !torch.int -> !torch.vtensor<[576,1,1],f32>
    %int1_1465 = torch.constant.int 1
    %1036 = torch.aten.sub.Tensor %1021, %1033, %int1_1465 : !torch.vtensor<[1,576,14,14],f32>, !torch.vtensor<[576,1,1],f32>, !torch.int -> !torch.vtensor<[1,576,14,14],f32>
    %1037 = torch.aten.mul.Tensor %1036, %1035 : !torch.vtensor<[1,576,14,14],f32>, !torch.vtensor<[576,1,1],f32> -> !torch.vtensor<[1,576,14,14],f32>
    %int-1_1466 = torch.constant.int -1
    %1038 = torch.aten.unsqueeze %arg174, %int-1_1466 : !torch.vtensor<[576],f32>, !torch.int -> !torch.vtensor<[576,1],f32>
    %int-1_1467 = torch.constant.int -1
    %1039 = torch.aten.unsqueeze %1038, %int-1_1467 : !torch.vtensor<[576,1],f32>, !torch.int -> !torch.vtensor<[576,1,1],f32>
    %1040 = torch.aten.mul.Tensor %1037, %1039 : !torch.vtensor<[1,576,14,14],f32>, !torch.vtensor<[576,1,1],f32> -> !torch.vtensor<[1,576,14,14],f32>
    %int-1_1468 = torch.constant.int -1
    %1041 = torch.aten.unsqueeze %arg175, %int-1_1468 : !torch.vtensor<[576],f32>, !torch.int -> !torch.vtensor<[576,1],f32>
    %int-1_1469 = torch.constant.int -1
    %1042 = torch.aten.unsqueeze %1041, %int-1_1469 : !torch.vtensor<[576,1],f32>, !torch.int -> !torch.vtensor<[576,1,1],f32>
    %int1_1470 = torch.constant.int 1
    %1043 = torch.aten.add.Tensor %1040, %1042, %int1_1470 : !torch.vtensor<[1,576,14,14],f32>, !torch.vtensor<[576,1,1],f32>, !torch.int -> !torch.vtensor<[1,576,14,14],f32>
    %float0.000000e00_1471 = torch.constant.float 0.000000e+00
    %float6.000000e00_1472 = torch.constant.float 6.000000e+00
    %1044 = torch.aten.hardtanh %1043, %float0.000000e00_1471, %float6.000000e00_1472 : !torch.vtensor<[1,576,14,14],f32>, !torch.float, !torch.float -> !torch.vtensor<[1,576,14,14],f32>
    %int0_1473 = torch.constant.int 0
    %int0_1474 = torch.constant.int 0
    %int0_1475 = torch.constant.int 0
    %int0_1476 = torch.constant.int 0
    %1045 = torch.prim.ListConstruct %int0_1473, %int0_1474, %int0_1475, %int0_1476 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %float0.000000e00_1477 = torch.constant.float 0.000000e+00
    %1046 = torch.aten.constant_pad_nd %1044, %1045, %float0.000000e00_1477 : !torch.vtensor<[1,576,14,14],f32>, !torch.list<int>, !torch.float -> !torch.vtensor<[1,576,14,14],f32>
    %none_1478 = torch.constant.none
    %int1_1479 = torch.constant.int 1
    %int1_1480 = torch.constant.int 1
    %1047 = torch.prim.ListConstruct %int1_1479, %int1_1480 : (!torch.int, !torch.int) -> !torch.list<int>
    %int0_1481 = torch.constant.int 0
    %int0_1482 = torch.constant.int 0
    %1048 = torch.prim.ListConstruct %int0_1481, %int0_1482 : (!torch.int, !torch.int) -> !torch.list<int>
    %int1_1483 = torch.constant.int 1
    %int1_1484 = torch.constant.int 1
    %1049 = torch.prim.ListConstruct %int1_1483, %int1_1484 : (!torch.int, !torch.int) -> !torch.list<int>
    %false_1485 = torch.constant.bool false
    %int0_1486 = torch.constant.int 0
    %int0_1487 = torch.constant.int 0
    %1050 = torch.prim.ListConstruct %int0_1486, %int0_1487 : (!torch.int, !torch.int) -> !torch.list<int>
    %int1_1488 = torch.constant.int 1
    %1051 = torch.aten.convolution %1046, %arg176, %none_1478, %1047, %1048, %1049, %false_1485, %1050, %int1_1488 : !torch.vtensor<[1,576,14,14],f32>, !torch.vtensor<[96,576,1,1],f32>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[1,96,14,14],f32>
    %int6_1489 = torch.constant.int 6
    %1052 = torch.prims.convert_element_type %arg177, %int6_1489 : !torch.vtensor<[96],f32>, !torch.int -> !torch.vtensor<[96],f32>
    %int6_1490 = torch.constant.int 6
    %1053 = torch.prims.convert_element_type %arg178, %int6_1490 : !torch.vtensor<[96],f32>, !torch.int -> !torch.vtensor<[96],f32>
    %float1.000000e-03_1491 = torch.constant.float 1.000000e-03
    %int1_1492 = torch.constant.int 1
    %1054 = torch.aten.add.Scalar %1053, %float1.000000e-03_1491, %int1_1492 : !torch.vtensor<[96],f32>, !torch.float, !torch.int -> !torch.vtensor<[96],f32>
    %1055 = torch.aten.sqrt %1054 : !torch.vtensor<[96],f32> -> !torch.vtensor<[96],f32>
    %1056 = torch.aten.reciprocal %1055 : !torch.vtensor<[96],f32> -> !torch.vtensor<[96],f32>
    %int1_1493 = torch.constant.int 1
    %1057 = torch.aten.mul.Scalar %1056, %int1_1493 : !torch.vtensor<[96],f32>, !torch.int -> !torch.vtensor<[96],f32>
    %int0_1494 = torch.constant.int 0
    %1058 = torch.prim.ListConstruct %int0_1494 : (!torch.int) -> !torch.list<int>
    %none_1495 = torch.constant.none
    %none_1496 = torch.constant.none
    %none_1497 = torch.constant.none
    %false_1498 = torch.constant.bool false
    %1059 = torch.aten.new_zeros %1051, %1058, %none_1495, %none_1496, %none_1497, %false_1498 : !torch.vtensor<[1,96,14,14],f32>, !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[0],f32>
    %int0_1499 = torch.constant.int 0
    %1060 = torch.prim.ListConstruct %int0_1499 : (!torch.int) -> !torch.list<int>
    %none_1500 = torch.constant.none
    %none_1501 = torch.constant.none
    %none_1502 = torch.constant.none
    %false_1503 = torch.constant.bool false
    %1061 = torch.aten.new_zeros %1051, %1060, %none_1500, %none_1501, %none_1502, %false_1503 : !torch.vtensor<[1,96,14,14],f32>, !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[0],f32>
    %int-1_1504 = torch.constant.int -1
    %1062 = torch.aten.unsqueeze %1052, %int-1_1504 : !torch.vtensor<[96],f32>, !torch.int -> !torch.vtensor<[96,1],f32>
    %int-1_1505 = torch.constant.int -1
    %1063 = torch.aten.unsqueeze %1062, %int-1_1505 : !torch.vtensor<[96,1],f32>, !torch.int -> !torch.vtensor<[96,1,1],f32>
    %int-1_1506 = torch.constant.int -1
    %1064 = torch.aten.unsqueeze %1057, %int-1_1506 : !torch.vtensor<[96],f32>, !torch.int -> !torch.vtensor<[96,1],f32>
    %int-1_1507 = torch.constant.int -1
    %1065 = torch.aten.unsqueeze %1064, %int-1_1507 : !torch.vtensor<[96,1],f32>, !torch.int -> !torch.vtensor<[96,1,1],f32>
    %int1_1508 = torch.constant.int 1
    %1066 = torch.aten.sub.Tensor %1051, %1063, %int1_1508 : !torch.vtensor<[1,96,14,14],f32>, !torch.vtensor<[96,1,1],f32>, !torch.int -> !torch.vtensor<[1,96,14,14],f32>
    %1067 = torch.aten.mul.Tensor %1066, %1065 : !torch.vtensor<[1,96,14,14],f32>, !torch.vtensor<[96,1,1],f32> -> !torch.vtensor<[1,96,14,14],f32>
    %int-1_1509 = torch.constant.int -1
    %1068 = torch.aten.unsqueeze %arg179, %int-1_1509 : !torch.vtensor<[96],f32>, !torch.int -> !torch.vtensor<[96,1],f32>
    %int-1_1510 = torch.constant.int -1
    %1069 = torch.aten.unsqueeze %1068, %int-1_1510 : !torch.vtensor<[96,1],f32>, !torch.int -> !torch.vtensor<[96,1,1],f32>
    %1070 = torch.aten.mul.Tensor %1067, %1069 : !torch.vtensor<[1,96,14,14],f32>, !torch.vtensor<[96,1,1],f32> -> !torch.vtensor<[1,96,14,14],f32>
    %int-1_1511 = torch.constant.int -1
    %1071 = torch.aten.unsqueeze %arg180, %int-1_1511 : !torch.vtensor<[96],f32>, !torch.int -> !torch.vtensor<[96,1],f32>
    %int-1_1512 = torch.constant.int -1
    %1072 = torch.aten.unsqueeze %1071, %int-1_1512 : !torch.vtensor<[96,1],f32>, !torch.int -> !torch.vtensor<[96,1,1],f32>
    %int1_1513 = torch.constant.int 1
    %1073 = torch.aten.add.Tensor %1070, %1072, %int1_1513 : !torch.vtensor<[1,96,14,14],f32>, !torch.vtensor<[96,1,1],f32>, !torch.int -> !torch.vtensor<[1,96,14,14],f32>
    %int1_1514 = torch.constant.int 1
    %1074 = torch.aten.add.Tensor %984, %1073, %int1_1514 : !torch.vtensor<[1,96,14,14],f32>, !torch.vtensor<[1,96,14,14],f32>, !torch.int -> !torch.vtensor<[1,96,14,14],f32>
    %int0_1515 = torch.constant.int 0
    %int0_1516 = torch.constant.int 0
    %int0_1517 = torch.constant.int 0
    %int0_1518 = torch.constant.int 0
    %1075 = torch.prim.ListConstruct %int0_1515, %int0_1516, %int0_1517, %int0_1518 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %float0.000000e00_1519 = torch.constant.float 0.000000e+00
    %1076 = torch.aten.constant_pad_nd %1074, %1075, %float0.000000e00_1519 : !torch.vtensor<[1,96,14,14],f32>, !torch.list<int>, !torch.float -> !torch.vtensor<[1,96,14,14],f32>
    %none_1520 = torch.constant.none
    %int1_1521 = torch.constant.int 1
    %int1_1522 = torch.constant.int 1
    %1077 = torch.prim.ListConstruct %int1_1521, %int1_1522 : (!torch.int, !torch.int) -> !torch.list<int>
    %int0_1523 = torch.constant.int 0
    %int0_1524 = torch.constant.int 0
    %1078 = torch.prim.ListConstruct %int0_1523, %int0_1524 : (!torch.int, !torch.int) -> !torch.list<int>
    %int1_1525 = torch.constant.int 1
    %int1_1526 = torch.constant.int 1
    %1079 = torch.prim.ListConstruct %int1_1525, %int1_1526 : (!torch.int, !torch.int) -> !torch.list<int>
    %false_1527 = torch.constant.bool false
    %int0_1528 = torch.constant.int 0
    %int0_1529 = torch.constant.int 0
    %1080 = torch.prim.ListConstruct %int0_1528, %int0_1529 : (!torch.int, !torch.int) -> !torch.list<int>
    %int1_1530 = torch.constant.int 1
    %1081 = torch.aten.convolution %1076, %arg181, %none_1520, %1077, %1078, %1079, %false_1527, %1080, %int1_1530 : !torch.vtensor<[1,96,14,14],f32>, !torch.vtensor<[576,96,1,1],f32>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[1,576,14,14],f32>
    %int6_1531 = torch.constant.int 6
    %1082 = torch.prims.convert_element_type %arg182, %int6_1531 : !torch.vtensor<[576],f32>, !torch.int -> !torch.vtensor<[576],f32>
    %int6_1532 = torch.constant.int 6
    %1083 = torch.prims.convert_element_type %arg183, %int6_1532 : !torch.vtensor<[576],f32>, !torch.int -> !torch.vtensor<[576],f32>
    %float1.000000e-03_1533 = torch.constant.float 1.000000e-03
    %int1_1534 = torch.constant.int 1
    %1084 = torch.aten.add.Scalar %1083, %float1.000000e-03_1533, %int1_1534 : !torch.vtensor<[576],f32>, !torch.float, !torch.int -> !torch.vtensor<[576],f32>
    %1085 = torch.aten.sqrt %1084 : !torch.vtensor<[576],f32> -> !torch.vtensor<[576],f32>
    %1086 = torch.aten.reciprocal %1085 : !torch.vtensor<[576],f32> -> !torch.vtensor<[576],f32>
    %int1_1535 = torch.constant.int 1
    %1087 = torch.aten.mul.Scalar %1086, %int1_1535 : !torch.vtensor<[576],f32>, !torch.int -> !torch.vtensor<[576],f32>
    %int0_1536 = torch.constant.int 0
    %1088 = torch.prim.ListConstruct %int0_1536 : (!torch.int) -> !torch.list<int>
    %none_1537 = torch.constant.none
    %none_1538 = torch.constant.none
    %none_1539 = torch.constant.none
    %false_1540 = torch.constant.bool false
    %1089 = torch.aten.new_zeros %1081, %1088, %none_1537, %none_1538, %none_1539, %false_1540 : !torch.vtensor<[1,576,14,14],f32>, !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[0],f32>
    %int0_1541 = torch.constant.int 0
    %1090 = torch.prim.ListConstruct %int0_1541 : (!torch.int) -> !torch.list<int>
    %none_1542 = torch.constant.none
    %none_1543 = torch.constant.none
    %none_1544 = torch.constant.none
    %false_1545 = torch.constant.bool false
    %1091 = torch.aten.new_zeros %1081, %1090, %none_1542, %none_1543, %none_1544, %false_1545 : !torch.vtensor<[1,576,14,14],f32>, !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[0],f32>
    %int-1_1546 = torch.constant.int -1
    %1092 = torch.aten.unsqueeze %1082, %int-1_1546 : !torch.vtensor<[576],f32>, !torch.int -> !torch.vtensor<[576,1],f32>
    %int-1_1547 = torch.constant.int -1
    %1093 = torch.aten.unsqueeze %1092, %int-1_1547 : !torch.vtensor<[576,1],f32>, !torch.int -> !torch.vtensor<[576,1,1],f32>
    %int-1_1548 = torch.constant.int -1
    %1094 = torch.aten.unsqueeze %1087, %int-1_1548 : !torch.vtensor<[576],f32>, !torch.int -> !torch.vtensor<[576,1],f32>
    %int-1_1549 = torch.constant.int -1
    %1095 = torch.aten.unsqueeze %1094, %int-1_1549 : !torch.vtensor<[576,1],f32>, !torch.int -> !torch.vtensor<[576,1,1],f32>
    %int1_1550 = torch.constant.int 1
    %1096 = torch.aten.sub.Tensor %1081, %1093, %int1_1550 : !torch.vtensor<[1,576,14,14],f32>, !torch.vtensor<[576,1,1],f32>, !torch.int -> !torch.vtensor<[1,576,14,14],f32>
    %1097 = torch.aten.mul.Tensor %1096, %1095 : !torch.vtensor<[1,576,14,14],f32>, !torch.vtensor<[576,1,1],f32> -> !torch.vtensor<[1,576,14,14],f32>
    %int-1_1551 = torch.constant.int -1
    %1098 = torch.aten.unsqueeze %arg184, %int-1_1551 : !torch.vtensor<[576],f32>, !torch.int -> !torch.vtensor<[576,1],f32>
    %int-1_1552 = torch.constant.int -1
    %1099 = torch.aten.unsqueeze %1098, %int-1_1552 : !torch.vtensor<[576,1],f32>, !torch.int -> !torch.vtensor<[576,1,1],f32>
    %1100 = torch.aten.mul.Tensor %1097, %1099 : !torch.vtensor<[1,576,14,14],f32>, !torch.vtensor<[576,1,1],f32> -> !torch.vtensor<[1,576,14,14],f32>
    %int-1_1553 = torch.constant.int -1
    %1101 = torch.aten.unsqueeze %arg185, %int-1_1553 : !torch.vtensor<[576],f32>, !torch.int -> !torch.vtensor<[576,1],f32>
    %int-1_1554 = torch.constant.int -1
    %1102 = torch.aten.unsqueeze %1101, %int-1_1554 : !torch.vtensor<[576,1],f32>, !torch.int -> !torch.vtensor<[576,1,1],f32>
    %int1_1555 = torch.constant.int 1
    %1103 = torch.aten.add.Tensor %1100, %1102, %int1_1555 : !torch.vtensor<[1,576,14,14],f32>, !torch.vtensor<[576,1,1],f32>, !torch.int -> !torch.vtensor<[1,576,14,14],f32>
    %float0.000000e00_1556 = torch.constant.float 0.000000e+00
    %float6.000000e00_1557 = torch.constant.float 6.000000e+00
    %1104 = torch.aten.hardtanh %1103, %float0.000000e00_1556, %float6.000000e00_1557 : !torch.vtensor<[1,576,14,14],f32>, !torch.float, !torch.float -> !torch.vtensor<[1,576,14,14],f32>
    %int1_1558 = torch.constant.int 1
    %int1_1559 = torch.constant.int 1
    %int1_1560 = torch.constant.int 1
    %int1_1561 = torch.constant.int 1
    %1105 = torch.prim.ListConstruct %int1_1558, %int1_1559, %int1_1560, %int1_1561 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %float0.000000e00_1562 = torch.constant.float 0.000000e+00
    %1106 = torch.aten.constant_pad_nd %1104, %1105, %float0.000000e00_1562 : !torch.vtensor<[1,576,14,14],f32>, !torch.list<int>, !torch.float -> !torch.vtensor<[1,576,16,16],f32>
    %none_1563 = torch.constant.none
    %int1_1564 = torch.constant.int 1
    %int1_1565 = torch.constant.int 1
    %1107 = torch.prim.ListConstruct %int1_1564, %int1_1565 : (!torch.int, !torch.int) -> !torch.list<int>
    %int0_1566 = torch.constant.int 0
    %int0_1567 = torch.constant.int 0
    %1108 = torch.prim.ListConstruct %int0_1566, %int0_1567 : (!torch.int, !torch.int) -> !torch.list<int>
    %int1_1568 = torch.constant.int 1
    %int1_1569 = torch.constant.int 1
    %1109 = torch.prim.ListConstruct %int1_1568, %int1_1569 : (!torch.int, !torch.int) -> !torch.list<int>
    %false_1570 = torch.constant.bool false
    %int0_1571 = torch.constant.int 0
    %int0_1572 = torch.constant.int 0
    %1110 = torch.prim.ListConstruct %int0_1571, %int0_1572 : (!torch.int, !torch.int) -> !torch.list<int>
    %int576_1573 = torch.constant.int 576
    %1111 = torch.aten.convolution %1106, %arg186, %none_1563, %1107, %1108, %1109, %false_1570, %1110, %int576_1573 : !torch.vtensor<[1,576,16,16],f32>, !torch.vtensor<[576,1,3,3],f32>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[1,576,14,14],f32>
    %int6_1574 = torch.constant.int 6
    %1112 = torch.prims.convert_element_type %arg187, %int6_1574 : !torch.vtensor<[576],f32>, !torch.int -> !torch.vtensor<[576],f32>
    %int6_1575 = torch.constant.int 6
    %1113 = torch.prims.convert_element_type %arg188, %int6_1575 : !torch.vtensor<[576],f32>, !torch.int -> !torch.vtensor<[576],f32>
    %float1.000000e-03_1576 = torch.constant.float 1.000000e-03
    %int1_1577 = torch.constant.int 1
    %1114 = torch.aten.add.Scalar %1113, %float1.000000e-03_1576, %int1_1577 : !torch.vtensor<[576],f32>, !torch.float, !torch.int -> !torch.vtensor<[576],f32>
    %1115 = torch.aten.sqrt %1114 : !torch.vtensor<[576],f32> -> !torch.vtensor<[576],f32>
    %1116 = torch.aten.reciprocal %1115 : !torch.vtensor<[576],f32> -> !torch.vtensor<[576],f32>
    %int1_1578 = torch.constant.int 1
    %1117 = torch.aten.mul.Scalar %1116, %int1_1578 : !torch.vtensor<[576],f32>, !torch.int -> !torch.vtensor<[576],f32>
    %int0_1579 = torch.constant.int 0
    %1118 = torch.prim.ListConstruct %int0_1579 : (!torch.int) -> !torch.list<int>
    %none_1580 = torch.constant.none
    %none_1581 = torch.constant.none
    %none_1582 = torch.constant.none
    %false_1583 = torch.constant.bool false
    %1119 = torch.aten.new_zeros %1111, %1118, %none_1580, %none_1581, %none_1582, %false_1583 : !torch.vtensor<[1,576,14,14],f32>, !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[0],f32>
    %int0_1584 = torch.constant.int 0
    %1120 = torch.prim.ListConstruct %int0_1584 : (!torch.int) -> !torch.list<int>
    %none_1585 = torch.constant.none
    %none_1586 = torch.constant.none
    %none_1587 = torch.constant.none
    %false_1588 = torch.constant.bool false
    %1121 = torch.aten.new_zeros %1111, %1120, %none_1585, %none_1586, %none_1587, %false_1588 : !torch.vtensor<[1,576,14,14],f32>, !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[0],f32>
    %int-1_1589 = torch.constant.int -1
    %1122 = torch.aten.unsqueeze %1112, %int-1_1589 : !torch.vtensor<[576],f32>, !torch.int -> !torch.vtensor<[576,1],f32>
    %int-1_1590 = torch.constant.int -1
    %1123 = torch.aten.unsqueeze %1122, %int-1_1590 : !torch.vtensor<[576,1],f32>, !torch.int -> !torch.vtensor<[576,1,1],f32>
    %int-1_1591 = torch.constant.int -1
    %1124 = torch.aten.unsqueeze %1117, %int-1_1591 : !torch.vtensor<[576],f32>, !torch.int -> !torch.vtensor<[576,1],f32>
    %int-1_1592 = torch.constant.int -1
    %1125 = torch.aten.unsqueeze %1124, %int-1_1592 : !torch.vtensor<[576,1],f32>, !torch.int -> !torch.vtensor<[576,1,1],f32>
    %int1_1593 = torch.constant.int 1
    %1126 = torch.aten.sub.Tensor %1111, %1123, %int1_1593 : !torch.vtensor<[1,576,14,14],f32>, !torch.vtensor<[576,1,1],f32>, !torch.int -> !torch.vtensor<[1,576,14,14],f32>
    %1127 = torch.aten.mul.Tensor %1126, %1125 : !torch.vtensor<[1,576,14,14],f32>, !torch.vtensor<[576,1,1],f32> -> !torch.vtensor<[1,576,14,14],f32>
    %int-1_1594 = torch.constant.int -1
    %1128 = torch.aten.unsqueeze %arg189, %int-1_1594 : !torch.vtensor<[576],f32>, !torch.int -> !torch.vtensor<[576,1],f32>
    %int-1_1595 = torch.constant.int -1
    %1129 = torch.aten.unsqueeze %1128, %int-1_1595 : !torch.vtensor<[576,1],f32>, !torch.int -> !torch.vtensor<[576,1,1],f32>
    %1130 = torch.aten.mul.Tensor %1127, %1129 : !torch.vtensor<[1,576,14,14],f32>, !torch.vtensor<[576,1,1],f32> -> !torch.vtensor<[1,576,14,14],f32>
    %int-1_1596 = torch.constant.int -1
    %1131 = torch.aten.unsqueeze %arg190, %int-1_1596 : !torch.vtensor<[576],f32>, !torch.int -> !torch.vtensor<[576,1],f32>
    %int-1_1597 = torch.constant.int -1
    %1132 = torch.aten.unsqueeze %1131, %int-1_1597 : !torch.vtensor<[576,1],f32>, !torch.int -> !torch.vtensor<[576,1,1],f32>
    %int1_1598 = torch.constant.int 1
    %1133 = torch.aten.add.Tensor %1130, %1132, %int1_1598 : !torch.vtensor<[1,576,14,14],f32>, !torch.vtensor<[576,1,1],f32>, !torch.int -> !torch.vtensor<[1,576,14,14],f32>
    %float0.000000e00_1599 = torch.constant.float 0.000000e+00
    %float6.000000e00_1600 = torch.constant.float 6.000000e+00
    %1134 = torch.aten.hardtanh %1133, %float0.000000e00_1599, %float6.000000e00_1600 : !torch.vtensor<[1,576,14,14],f32>, !torch.float, !torch.float -> !torch.vtensor<[1,576,14,14],f32>
    %int0_1601 = torch.constant.int 0
    %int0_1602 = torch.constant.int 0
    %int0_1603 = torch.constant.int 0
    %int0_1604 = torch.constant.int 0
    %1135 = torch.prim.ListConstruct %int0_1601, %int0_1602, %int0_1603, %int0_1604 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %float0.000000e00_1605 = torch.constant.float 0.000000e+00
    %1136 = torch.aten.constant_pad_nd %1134, %1135, %float0.000000e00_1605 : !torch.vtensor<[1,576,14,14],f32>, !torch.list<int>, !torch.float -> !torch.vtensor<[1,576,14,14],f32>
    %none_1606 = torch.constant.none
    %int1_1607 = torch.constant.int 1
    %int1_1608 = torch.constant.int 1
    %1137 = torch.prim.ListConstruct %int1_1607, %int1_1608 : (!torch.int, !torch.int) -> !torch.list<int>
    %int0_1609 = torch.constant.int 0
    %int0_1610 = torch.constant.int 0
    %1138 = torch.prim.ListConstruct %int0_1609, %int0_1610 : (!torch.int, !torch.int) -> !torch.list<int>
    %int1_1611 = torch.constant.int 1
    %int1_1612 = torch.constant.int 1
    %1139 = torch.prim.ListConstruct %int1_1611, %int1_1612 : (!torch.int, !torch.int) -> !torch.list<int>
    %false_1613 = torch.constant.bool false
    %int0_1614 = torch.constant.int 0
    %int0_1615 = torch.constant.int 0
    %1140 = torch.prim.ListConstruct %int0_1614, %int0_1615 : (!torch.int, !torch.int) -> !torch.list<int>
    %int1_1616 = torch.constant.int 1
    %1141 = torch.aten.convolution %1136, %arg191, %none_1606, %1137, %1138, %1139, %false_1613, %1140, %int1_1616 : !torch.vtensor<[1,576,14,14],f32>, !torch.vtensor<[96,576,1,1],f32>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[1,96,14,14],f32>
    %int6_1617 = torch.constant.int 6
    %1142 = torch.prims.convert_element_type %arg192, %int6_1617 : !torch.vtensor<[96],f32>, !torch.int -> !torch.vtensor<[96],f32>
    %int6_1618 = torch.constant.int 6
    %1143 = torch.prims.convert_element_type %arg193, %int6_1618 : !torch.vtensor<[96],f32>, !torch.int -> !torch.vtensor<[96],f32>
    %float1.000000e-03_1619 = torch.constant.float 1.000000e-03
    %int1_1620 = torch.constant.int 1
    %1144 = torch.aten.add.Scalar %1143, %float1.000000e-03_1619, %int1_1620 : !torch.vtensor<[96],f32>, !torch.float, !torch.int -> !torch.vtensor<[96],f32>
    %1145 = torch.aten.sqrt %1144 : !torch.vtensor<[96],f32> -> !torch.vtensor<[96],f32>
    %1146 = torch.aten.reciprocal %1145 : !torch.vtensor<[96],f32> -> !torch.vtensor<[96],f32>
    %int1_1621 = torch.constant.int 1
    %1147 = torch.aten.mul.Scalar %1146, %int1_1621 : !torch.vtensor<[96],f32>, !torch.int -> !torch.vtensor<[96],f32>
    %int0_1622 = torch.constant.int 0
    %1148 = torch.prim.ListConstruct %int0_1622 : (!torch.int) -> !torch.list<int>
    %none_1623 = torch.constant.none
    %none_1624 = torch.constant.none
    %none_1625 = torch.constant.none
    %false_1626 = torch.constant.bool false
    %1149 = torch.aten.new_zeros %1141, %1148, %none_1623, %none_1624, %none_1625, %false_1626 : !torch.vtensor<[1,96,14,14],f32>, !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[0],f32>
    %int0_1627 = torch.constant.int 0
    %1150 = torch.prim.ListConstruct %int0_1627 : (!torch.int) -> !torch.list<int>
    %none_1628 = torch.constant.none
    %none_1629 = torch.constant.none
    %none_1630 = torch.constant.none
    %false_1631 = torch.constant.bool false
    %1151 = torch.aten.new_zeros %1141, %1150, %none_1628, %none_1629, %none_1630, %false_1631 : !torch.vtensor<[1,96,14,14],f32>, !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[0],f32>
    %int-1_1632 = torch.constant.int -1
    %1152 = torch.aten.unsqueeze %1142, %int-1_1632 : !torch.vtensor<[96],f32>, !torch.int -> !torch.vtensor<[96,1],f32>
    %int-1_1633 = torch.constant.int -1
    %1153 = torch.aten.unsqueeze %1152, %int-1_1633 : !torch.vtensor<[96,1],f32>, !torch.int -> !torch.vtensor<[96,1,1],f32>
    %int-1_1634 = torch.constant.int -1
    %1154 = torch.aten.unsqueeze %1147, %int-1_1634 : !torch.vtensor<[96],f32>, !torch.int -> !torch.vtensor<[96,1],f32>
    %int-1_1635 = torch.constant.int -1
    %1155 = torch.aten.unsqueeze %1154, %int-1_1635 : !torch.vtensor<[96,1],f32>, !torch.int -> !torch.vtensor<[96,1,1],f32>
    %int1_1636 = torch.constant.int 1
    %1156 = torch.aten.sub.Tensor %1141, %1153, %int1_1636 : !torch.vtensor<[1,96,14,14],f32>, !torch.vtensor<[96,1,1],f32>, !torch.int -> !torch.vtensor<[1,96,14,14],f32>
    %1157 = torch.aten.mul.Tensor %1156, %1155 : !torch.vtensor<[1,96,14,14],f32>, !torch.vtensor<[96,1,1],f32> -> !torch.vtensor<[1,96,14,14],f32>
    %int-1_1637 = torch.constant.int -1
    %1158 = torch.aten.unsqueeze %arg194, %int-1_1637 : !torch.vtensor<[96],f32>, !torch.int -> !torch.vtensor<[96,1],f32>
    %int-1_1638 = torch.constant.int -1
    %1159 = torch.aten.unsqueeze %1158, %int-1_1638 : !torch.vtensor<[96,1],f32>, !torch.int -> !torch.vtensor<[96,1,1],f32>
    %1160 = torch.aten.mul.Tensor %1157, %1159 : !torch.vtensor<[1,96,14,14],f32>, !torch.vtensor<[96,1,1],f32> -> !torch.vtensor<[1,96,14,14],f32>
    %int-1_1639 = torch.constant.int -1
    %1161 = torch.aten.unsqueeze %arg195, %int-1_1639 : !torch.vtensor<[96],f32>, !torch.int -> !torch.vtensor<[96,1],f32>
    %int-1_1640 = torch.constant.int -1
    %1162 = torch.aten.unsqueeze %1161, %int-1_1640 : !torch.vtensor<[96,1],f32>, !torch.int -> !torch.vtensor<[96,1,1],f32>
    %int1_1641 = torch.constant.int 1
    %1163 = torch.aten.add.Tensor %1160, %1162, %int1_1641 : !torch.vtensor<[1,96,14,14],f32>, !torch.vtensor<[96,1,1],f32>, !torch.int -> !torch.vtensor<[1,96,14,14],f32>
    %int1_1642 = torch.constant.int 1
    %1164 = torch.aten.add.Tensor %1074, %1163, %int1_1642 : !torch.vtensor<[1,96,14,14],f32>, !torch.vtensor<[1,96,14,14],f32>, !torch.int -> !torch.vtensor<[1,96,14,14],f32>
    %int0_1643 = torch.constant.int 0
    %int0_1644 = torch.constant.int 0
    %int0_1645 = torch.constant.int 0
    %int0_1646 = torch.constant.int 0
    %1165 = torch.prim.ListConstruct %int0_1643, %int0_1644, %int0_1645, %int0_1646 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %float0.000000e00_1647 = torch.constant.float 0.000000e+00
    %1166 = torch.aten.constant_pad_nd %1164, %1165, %float0.000000e00_1647 : !torch.vtensor<[1,96,14,14],f32>, !torch.list<int>, !torch.float -> !torch.vtensor<[1,96,14,14],f32>
    %none_1648 = torch.constant.none
    %int1_1649 = torch.constant.int 1
    %int1_1650 = torch.constant.int 1
    %1167 = torch.prim.ListConstruct %int1_1649, %int1_1650 : (!torch.int, !torch.int) -> !torch.list<int>
    %int0_1651 = torch.constant.int 0
    %int0_1652 = torch.constant.int 0
    %1168 = torch.prim.ListConstruct %int0_1651, %int0_1652 : (!torch.int, !torch.int) -> !torch.list<int>
    %int1_1653 = torch.constant.int 1
    %int1_1654 = torch.constant.int 1
    %1169 = torch.prim.ListConstruct %int1_1653, %int1_1654 : (!torch.int, !torch.int) -> !torch.list<int>
    %false_1655 = torch.constant.bool false
    %int0_1656 = torch.constant.int 0
    %int0_1657 = torch.constant.int 0
    %1170 = torch.prim.ListConstruct %int0_1656, %int0_1657 : (!torch.int, !torch.int) -> !torch.list<int>
    %int1_1658 = torch.constant.int 1
    %1171 = torch.aten.convolution %1166, %arg196, %none_1648, %1167, %1168, %1169, %false_1655, %1170, %int1_1658 : !torch.vtensor<[1,96,14,14],f32>, !torch.vtensor<[576,96,1,1],f32>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[1,576,14,14],f32>
    %int6_1659 = torch.constant.int 6
    %1172 = torch.prims.convert_element_type %arg197, %int6_1659 : !torch.vtensor<[576],f32>, !torch.int -> !torch.vtensor<[576],f32>
    %int6_1660 = torch.constant.int 6
    %1173 = torch.prims.convert_element_type %arg198, %int6_1660 : !torch.vtensor<[576],f32>, !torch.int -> !torch.vtensor<[576],f32>
    %float1.000000e-03_1661 = torch.constant.float 1.000000e-03
    %int1_1662 = torch.constant.int 1
    %1174 = torch.aten.add.Scalar %1173, %float1.000000e-03_1661, %int1_1662 : !torch.vtensor<[576],f32>, !torch.float, !torch.int -> !torch.vtensor<[576],f32>
    %1175 = torch.aten.sqrt %1174 : !torch.vtensor<[576],f32> -> !torch.vtensor<[576],f32>
    %1176 = torch.aten.reciprocal %1175 : !torch.vtensor<[576],f32> -> !torch.vtensor<[576],f32>
    %int1_1663 = torch.constant.int 1
    %1177 = torch.aten.mul.Scalar %1176, %int1_1663 : !torch.vtensor<[576],f32>, !torch.int -> !torch.vtensor<[576],f32>
    %int0_1664 = torch.constant.int 0
    %1178 = torch.prim.ListConstruct %int0_1664 : (!torch.int) -> !torch.list<int>
    %none_1665 = torch.constant.none
    %none_1666 = torch.constant.none
    %none_1667 = torch.constant.none
    %false_1668 = torch.constant.bool false
    %1179 = torch.aten.new_zeros %1171, %1178, %none_1665, %none_1666, %none_1667, %false_1668 : !torch.vtensor<[1,576,14,14],f32>, !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[0],f32>
    %int0_1669 = torch.constant.int 0
    %1180 = torch.prim.ListConstruct %int0_1669 : (!torch.int) -> !torch.list<int>
    %none_1670 = torch.constant.none
    %none_1671 = torch.constant.none
    %none_1672 = torch.constant.none
    %false_1673 = torch.constant.bool false
    %1181 = torch.aten.new_zeros %1171, %1180, %none_1670, %none_1671, %none_1672, %false_1673 : !torch.vtensor<[1,576,14,14],f32>, !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[0],f32>
    %int-1_1674 = torch.constant.int -1
    %1182 = torch.aten.unsqueeze %1172, %int-1_1674 : !torch.vtensor<[576],f32>, !torch.int -> !torch.vtensor<[576,1],f32>
    %int-1_1675 = torch.constant.int -1
    %1183 = torch.aten.unsqueeze %1182, %int-1_1675 : !torch.vtensor<[576,1],f32>, !torch.int -> !torch.vtensor<[576,1,1],f32>
    %int-1_1676 = torch.constant.int -1
    %1184 = torch.aten.unsqueeze %1177, %int-1_1676 : !torch.vtensor<[576],f32>, !torch.int -> !torch.vtensor<[576,1],f32>
    %int-1_1677 = torch.constant.int -1
    %1185 = torch.aten.unsqueeze %1184, %int-1_1677 : !torch.vtensor<[576,1],f32>, !torch.int -> !torch.vtensor<[576,1,1],f32>
    %int1_1678 = torch.constant.int 1
    %1186 = torch.aten.sub.Tensor %1171, %1183, %int1_1678 : !torch.vtensor<[1,576,14,14],f32>, !torch.vtensor<[576,1,1],f32>, !torch.int -> !torch.vtensor<[1,576,14,14],f32>
    %1187 = torch.aten.mul.Tensor %1186, %1185 : !torch.vtensor<[1,576,14,14],f32>, !torch.vtensor<[576,1,1],f32> -> !torch.vtensor<[1,576,14,14],f32>
    %int-1_1679 = torch.constant.int -1
    %1188 = torch.aten.unsqueeze %arg199, %int-1_1679 : !torch.vtensor<[576],f32>, !torch.int -> !torch.vtensor<[576,1],f32>
    %int-1_1680 = torch.constant.int -1
    %1189 = torch.aten.unsqueeze %1188, %int-1_1680 : !torch.vtensor<[576,1],f32>, !torch.int -> !torch.vtensor<[576,1,1],f32>
    %1190 = torch.aten.mul.Tensor %1187, %1189 : !torch.vtensor<[1,576,14,14],f32>, !torch.vtensor<[576,1,1],f32> -> !torch.vtensor<[1,576,14,14],f32>
    %int-1_1681 = torch.constant.int -1
    %1191 = torch.aten.unsqueeze %arg200, %int-1_1681 : !torch.vtensor<[576],f32>, !torch.int -> !torch.vtensor<[576,1],f32>
    %int-1_1682 = torch.constant.int -1
    %1192 = torch.aten.unsqueeze %1191, %int-1_1682 : !torch.vtensor<[576,1],f32>, !torch.int -> !torch.vtensor<[576,1,1],f32>
    %int1_1683 = torch.constant.int 1
    %1193 = torch.aten.add.Tensor %1190, %1192, %int1_1683 : !torch.vtensor<[1,576,14,14],f32>, !torch.vtensor<[576,1,1],f32>, !torch.int -> !torch.vtensor<[1,576,14,14],f32>
    %float0.000000e00_1684 = torch.constant.float 0.000000e+00
    %float6.000000e00_1685 = torch.constant.float 6.000000e+00
    %1194 = torch.aten.hardtanh %1193, %float0.000000e00_1684, %float6.000000e00_1685 : !torch.vtensor<[1,576,14,14],f32>, !torch.float, !torch.float -> !torch.vtensor<[1,576,14,14],f32>
    %int0_1686 = torch.constant.int 0
    %int1_1687 = torch.constant.int 1
    %int0_1688 = torch.constant.int 0
    %int1_1689 = torch.constant.int 1
    %1195 = torch.prim.ListConstruct %int0_1686, %int1_1687, %int0_1688, %int1_1689 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %float0.000000e00_1690 = torch.constant.float 0.000000e+00
    %1196 = torch.aten.constant_pad_nd %1194, %1195, %float0.000000e00_1690 : !torch.vtensor<[1,576,14,14],f32>, !torch.list<int>, !torch.float -> !torch.vtensor<[1,576,15,15],f32>
    %none_1691 = torch.constant.none
    %int2_1692 = torch.constant.int 2
    %int2_1693 = torch.constant.int 2
    %1197 = torch.prim.ListConstruct %int2_1692, %int2_1693 : (!torch.int, !torch.int) -> !torch.list<int>
    %int0_1694 = torch.constant.int 0
    %int0_1695 = torch.constant.int 0
    %1198 = torch.prim.ListConstruct %int0_1694, %int0_1695 : (!torch.int, !torch.int) -> !torch.list<int>
    %int1_1696 = torch.constant.int 1
    %int1_1697 = torch.constant.int 1
    %1199 = torch.prim.ListConstruct %int1_1696, %int1_1697 : (!torch.int, !torch.int) -> !torch.list<int>
    %false_1698 = torch.constant.bool false
    %int0_1699 = torch.constant.int 0
    %int0_1700 = torch.constant.int 0
    %1200 = torch.prim.ListConstruct %int0_1699, %int0_1700 : (!torch.int, !torch.int) -> !torch.list<int>
    %int576_1701 = torch.constant.int 576
    %1201 = torch.aten.convolution %1196, %arg201, %none_1691, %1197, %1198, %1199, %false_1698, %1200, %int576_1701 : !torch.vtensor<[1,576,15,15],f32>, !torch.vtensor<[576,1,3,3],f32>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[1,576,7,7],f32>
    %int6_1702 = torch.constant.int 6
    %1202 = torch.prims.convert_element_type %arg202, %int6_1702 : !torch.vtensor<[576],f32>, !torch.int -> !torch.vtensor<[576],f32>
    %int6_1703 = torch.constant.int 6
    %1203 = torch.prims.convert_element_type %arg203, %int6_1703 : !torch.vtensor<[576],f32>, !torch.int -> !torch.vtensor<[576],f32>
    %float1.000000e-03_1704 = torch.constant.float 1.000000e-03
    %int1_1705 = torch.constant.int 1
    %1204 = torch.aten.add.Scalar %1203, %float1.000000e-03_1704, %int1_1705 : !torch.vtensor<[576],f32>, !torch.float, !torch.int -> !torch.vtensor<[576],f32>
    %1205 = torch.aten.sqrt %1204 : !torch.vtensor<[576],f32> -> !torch.vtensor<[576],f32>
    %1206 = torch.aten.reciprocal %1205 : !torch.vtensor<[576],f32> -> !torch.vtensor<[576],f32>
    %int1_1706 = torch.constant.int 1
    %1207 = torch.aten.mul.Scalar %1206, %int1_1706 : !torch.vtensor<[576],f32>, !torch.int -> !torch.vtensor<[576],f32>
    %int0_1707 = torch.constant.int 0
    %1208 = torch.prim.ListConstruct %int0_1707 : (!torch.int) -> !torch.list<int>
    %none_1708 = torch.constant.none
    %none_1709 = torch.constant.none
    %none_1710 = torch.constant.none
    %false_1711 = torch.constant.bool false
    %1209 = torch.aten.new_zeros %1201, %1208, %none_1708, %none_1709, %none_1710, %false_1711 : !torch.vtensor<[1,576,7,7],f32>, !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[0],f32>
    %int0_1712 = torch.constant.int 0
    %1210 = torch.prim.ListConstruct %int0_1712 : (!torch.int) -> !torch.list<int>
    %none_1713 = torch.constant.none
    %none_1714 = torch.constant.none
    %none_1715 = torch.constant.none
    %false_1716 = torch.constant.bool false
    %1211 = torch.aten.new_zeros %1201, %1210, %none_1713, %none_1714, %none_1715, %false_1716 : !torch.vtensor<[1,576,7,7],f32>, !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[0],f32>
    %int-1_1717 = torch.constant.int -1
    %1212 = torch.aten.unsqueeze %1202, %int-1_1717 : !torch.vtensor<[576],f32>, !torch.int -> !torch.vtensor<[576,1],f32>
    %int-1_1718 = torch.constant.int -1
    %1213 = torch.aten.unsqueeze %1212, %int-1_1718 : !torch.vtensor<[576,1],f32>, !torch.int -> !torch.vtensor<[576,1,1],f32>
    %int-1_1719 = torch.constant.int -1
    %1214 = torch.aten.unsqueeze %1207, %int-1_1719 : !torch.vtensor<[576],f32>, !torch.int -> !torch.vtensor<[576,1],f32>
    %int-1_1720 = torch.constant.int -1
    %1215 = torch.aten.unsqueeze %1214, %int-1_1720 : !torch.vtensor<[576,1],f32>, !torch.int -> !torch.vtensor<[576,1,1],f32>
    %int1_1721 = torch.constant.int 1
    %1216 = torch.aten.sub.Tensor %1201, %1213, %int1_1721 : !torch.vtensor<[1,576,7,7],f32>, !torch.vtensor<[576,1,1],f32>, !torch.int -> !torch.vtensor<[1,576,7,7],f32>
    %1217 = torch.aten.mul.Tensor %1216, %1215 : !torch.vtensor<[1,576,7,7],f32>, !torch.vtensor<[576,1,1],f32> -> !torch.vtensor<[1,576,7,7],f32>
    %int-1_1722 = torch.constant.int -1
    %1218 = torch.aten.unsqueeze %arg204, %int-1_1722 : !torch.vtensor<[576],f32>, !torch.int -> !torch.vtensor<[576,1],f32>
    %int-1_1723 = torch.constant.int -1
    %1219 = torch.aten.unsqueeze %1218, %int-1_1723 : !torch.vtensor<[576,1],f32>, !torch.int -> !torch.vtensor<[576,1,1],f32>
    %1220 = torch.aten.mul.Tensor %1217, %1219 : !torch.vtensor<[1,576,7,7],f32>, !torch.vtensor<[576,1,1],f32> -> !torch.vtensor<[1,576,7,7],f32>
    %int-1_1724 = torch.constant.int -1
    %1221 = torch.aten.unsqueeze %arg205, %int-1_1724 : !torch.vtensor<[576],f32>, !torch.int -> !torch.vtensor<[576,1],f32>
    %int-1_1725 = torch.constant.int -1
    %1222 = torch.aten.unsqueeze %1221, %int-1_1725 : !torch.vtensor<[576,1],f32>, !torch.int -> !torch.vtensor<[576,1,1],f32>
    %int1_1726 = torch.constant.int 1
    %1223 = torch.aten.add.Tensor %1220, %1222, %int1_1726 : !torch.vtensor<[1,576,7,7],f32>, !torch.vtensor<[576,1,1],f32>, !torch.int -> !torch.vtensor<[1,576,7,7],f32>
    %float0.000000e00_1727 = torch.constant.float 0.000000e+00
    %float6.000000e00_1728 = torch.constant.float 6.000000e+00
    %1224 = torch.aten.hardtanh %1223, %float0.000000e00_1727, %float6.000000e00_1728 : !torch.vtensor<[1,576,7,7],f32>, !torch.float, !torch.float -> !torch.vtensor<[1,576,7,7],f32>
    %int0_1729 = torch.constant.int 0
    %int0_1730 = torch.constant.int 0
    %int0_1731 = torch.constant.int 0
    %int0_1732 = torch.constant.int 0
    %1225 = torch.prim.ListConstruct %int0_1729, %int0_1730, %int0_1731, %int0_1732 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %float0.000000e00_1733 = torch.constant.float 0.000000e+00
    %1226 = torch.aten.constant_pad_nd %1224, %1225, %float0.000000e00_1733 : !torch.vtensor<[1,576,7,7],f32>, !torch.list<int>, !torch.float -> !torch.vtensor<[1,576,7,7],f32>
    %none_1734 = torch.constant.none
    %int1_1735 = torch.constant.int 1
    %int1_1736 = torch.constant.int 1
    %1227 = torch.prim.ListConstruct %int1_1735, %int1_1736 : (!torch.int, !torch.int) -> !torch.list<int>
    %int0_1737 = torch.constant.int 0
    %int0_1738 = torch.constant.int 0
    %1228 = torch.prim.ListConstruct %int0_1737, %int0_1738 : (!torch.int, !torch.int) -> !torch.list<int>
    %int1_1739 = torch.constant.int 1
    %int1_1740 = torch.constant.int 1
    %1229 = torch.prim.ListConstruct %int1_1739, %int1_1740 : (!torch.int, !torch.int) -> !torch.list<int>
    %false_1741 = torch.constant.bool false
    %int0_1742 = torch.constant.int 0
    %int0_1743 = torch.constant.int 0
    %1230 = torch.prim.ListConstruct %int0_1742, %int0_1743 : (!torch.int, !torch.int) -> !torch.list<int>
    %int1_1744 = torch.constant.int 1
    %1231 = torch.aten.convolution %1226, %arg206, %none_1734, %1227, %1228, %1229, %false_1741, %1230, %int1_1744 : !torch.vtensor<[1,576,7,7],f32>, !torch.vtensor<[160,576,1,1],f32>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[1,160,7,7],f32>
    %int6_1745 = torch.constant.int 6
    %1232 = torch.prims.convert_element_type %arg207, %int6_1745 : !torch.vtensor<[160],f32>, !torch.int -> !torch.vtensor<[160],f32>
    %int6_1746 = torch.constant.int 6
    %1233 = torch.prims.convert_element_type %arg208, %int6_1746 : !torch.vtensor<[160],f32>, !torch.int -> !torch.vtensor<[160],f32>
    %float1.000000e-03_1747 = torch.constant.float 1.000000e-03
    %int1_1748 = torch.constant.int 1
    %1234 = torch.aten.add.Scalar %1233, %float1.000000e-03_1747, %int1_1748 : !torch.vtensor<[160],f32>, !torch.float, !torch.int -> !torch.vtensor<[160],f32>
    %1235 = torch.aten.sqrt %1234 : !torch.vtensor<[160],f32> -> !torch.vtensor<[160],f32>
    %1236 = torch.aten.reciprocal %1235 : !torch.vtensor<[160],f32> -> !torch.vtensor<[160],f32>
    %int1_1749 = torch.constant.int 1
    %1237 = torch.aten.mul.Scalar %1236, %int1_1749 : !torch.vtensor<[160],f32>, !torch.int -> !torch.vtensor<[160],f32>
    %int0_1750 = torch.constant.int 0
    %1238 = torch.prim.ListConstruct %int0_1750 : (!torch.int) -> !torch.list<int>
    %none_1751 = torch.constant.none
    %none_1752 = torch.constant.none
    %none_1753 = torch.constant.none
    %false_1754 = torch.constant.bool false
    %1239 = torch.aten.new_zeros %1231, %1238, %none_1751, %none_1752, %none_1753, %false_1754 : !torch.vtensor<[1,160,7,7],f32>, !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[0],f32>
    %int0_1755 = torch.constant.int 0
    %1240 = torch.prim.ListConstruct %int0_1755 : (!torch.int) -> !torch.list<int>
    %none_1756 = torch.constant.none
    %none_1757 = torch.constant.none
    %none_1758 = torch.constant.none
    %false_1759 = torch.constant.bool false
    %1241 = torch.aten.new_zeros %1231, %1240, %none_1756, %none_1757, %none_1758, %false_1759 : !torch.vtensor<[1,160,7,7],f32>, !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[0],f32>
    %int-1_1760 = torch.constant.int -1
    %1242 = torch.aten.unsqueeze %1232, %int-1_1760 : !torch.vtensor<[160],f32>, !torch.int -> !torch.vtensor<[160,1],f32>
    %int-1_1761 = torch.constant.int -1
    %1243 = torch.aten.unsqueeze %1242, %int-1_1761 : !torch.vtensor<[160,1],f32>, !torch.int -> !torch.vtensor<[160,1,1],f32>
    %int-1_1762 = torch.constant.int -1
    %1244 = torch.aten.unsqueeze %1237, %int-1_1762 : !torch.vtensor<[160],f32>, !torch.int -> !torch.vtensor<[160,1],f32>
    %int-1_1763 = torch.constant.int -1
    %1245 = torch.aten.unsqueeze %1244, %int-1_1763 : !torch.vtensor<[160,1],f32>, !torch.int -> !torch.vtensor<[160,1,1],f32>
    %int1_1764 = torch.constant.int 1
    %1246 = torch.aten.sub.Tensor %1231, %1243, %int1_1764 : !torch.vtensor<[1,160,7,7],f32>, !torch.vtensor<[160,1,1],f32>, !torch.int -> !torch.vtensor<[1,160,7,7],f32>
    %1247 = torch.aten.mul.Tensor %1246, %1245 : !torch.vtensor<[1,160,7,7],f32>, !torch.vtensor<[160,1,1],f32> -> !torch.vtensor<[1,160,7,7],f32>
    %int-1_1765 = torch.constant.int -1
    %1248 = torch.aten.unsqueeze %arg209, %int-1_1765 : !torch.vtensor<[160],f32>, !torch.int -> !torch.vtensor<[160,1],f32>
    %int-1_1766 = torch.constant.int -1
    %1249 = torch.aten.unsqueeze %1248, %int-1_1766 : !torch.vtensor<[160,1],f32>, !torch.int -> !torch.vtensor<[160,1,1],f32>
    %1250 = torch.aten.mul.Tensor %1247, %1249 : !torch.vtensor<[1,160,7,7],f32>, !torch.vtensor<[160,1,1],f32> -> !torch.vtensor<[1,160,7,7],f32>
    %int-1_1767 = torch.constant.int -1
    %1251 = torch.aten.unsqueeze %arg210, %int-1_1767 : !torch.vtensor<[160],f32>, !torch.int -> !torch.vtensor<[160,1],f32>
    %int-1_1768 = torch.constant.int -1
    %1252 = torch.aten.unsqueeze %1251, %int-1_1768 : !torch.vtensor<[160,1],f32>, !torch.int -> !torch.vtensor<[160,1,1],f32>
    %int1_1769 = torch.constant.int 1
    %1253 = torch.aten.add.Tensor %1250, %1252, %int1_1769 : !torch.vtensor<[1,160,7,7],f32>, !torch.vtensor<[160,1,1],f32>, !torch.int -> !torch.vtensor<[1,160,7,7],f32>
    %int0_1770 = torch.constant.int 0
    %int0_1771 = torch.constant.int 0
    %int0_1772 = torch.constant.int 0
    %int0_1773 = torch.constant.int 0
    %1254 = torch.prim.ListConstruct %int0_1770, %int0_1771, %int0_1772, %int0_1773 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %float0.000000e00_1774 = torch.constant.float 0.000000e+00
    %1255 = torch.aten.constant_pad_nd %1253, %1254, %float0.000000e00_1774 : !torch.vtensor<[1,160,7,7],f32>, !torch.list<int>, !torch.float -> !torch.vtensor<[1,160,7,7],f32>
    %none_1775 = torch.constant.none
    %int1_1776 = torch.constant.int 1
    %int1_1777 = torch.constant.int 1
    %1256 = torch.prim.ListConstruct %int1_1776, %int1_1777 : (!torch.int, !torch.int) -> !torch.list<int>
    %int0_1778 = torch.constant.int 0
    %int0_1779 = torch.constant.int 0
    %1257 = torch.prim.ListConstruct %int0_1778, %int0_1779 : (!torch.int, !torch.int) -> !torch.list<int>
    %int1_1780 = torch.constant.int 1
    %int1_1781 = torch.constant.int 1
    %1258 = torch.prim.ListConstruct %int1_1780, %int1_1781 : (!torch.int, !torch.int) -> !torch.list<int>
    %false_1782 = torch.constant.bool false
    %int0_1783 = torch.constant.int 0
    %int0_1784 = torch.constant.int 0
    %1259 = torch.prim.ListConstruct %int0_1783, %int0_1784 : (!torch.int, !torch.int) -> !torch.list<int>
    %int1_1785 = torch.constant.int 1
    %1260 = torch.aten.convolution %1255, %arg211, %none_1775, %1256, %1257, %1258, %false_1782, %1259, %int1_1785 : !torch.vtensor<[1,160,7,7],f32>, !torch.vtensor<[960,160,1,1],f32>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[1,960,7,7],f32>
    %int6_1786 = torch.constant.int 6
    %1261 = torch.prims.convert_element_type %arg212, %int6_1786 : !torch.vtensor<[960],f32>, !torch.int -> !torch.vtensor<[960],f32>
    %int6_1787 = torch.constant.int 6
    %1262 = torch.prims.convert_element_type %arg213, %int6_1787 : !torch.vtensor<[960],f32>, !torch.int -> !torch.vtensor<[960],f32>
    %float1.000000e-03_1788 = torch.constant.float 1.000000e-03
    %int1_1789 = torch.constant.int 1
    %1263 = torch.aten.add.Scalar %1262, %float1.000000e-03_1788, %int1_1789 : !torch.vtensor<[960],f32>, !torch.float, !torch.int -> !torch.vtensor<[960],f32>
    %1264 = torch.aten.sqrt %1263 : !torch.vtensor<[960],f32> -> !torch.vtensor<[960],f32>
    %1265 = torch.aten.reciprocal %1264 : !torch.vtensor<[960],f32> -> !torch.vtensor<[960],f32>
    %int1_1790 = torch.constant.int 1
    %1266 = torch.aten.mul.Scalar %1265, %int1_1790 : !torch.vtensor<[960],f32>, !torch.int -> !torch.vtensor<[960],f32>
    %int0_1791 = torch.constant.int 0
    %1267 = torch.prim.ListConstruct %int0_1791 : (!torch.int) -> !torch.list<int>
    %none_1792 = torch.constant.none
    %none_1793 = torch.constant.none
    %none_1794 = torch.constant.none
    %false_1795 = torch.constant.bool false
    %1268 = torch.aten.new_zeros %1260, %1267, %none_1792, %none_1793, %none_1794, %false_1795 : !torch.vtensor<[1,960,7,7],f32>, !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[0],f32>
    %int0_1796 = torch.constant.int 0
    %1269 = torch.prim.ListConstruct %int0_1796 : (!torch.int) -> !torch.list<int>
    %none_1797 = torch.constant.none
    %none_1798 = torch.constant.none
    %none_1799 = torch.constant.none
    %false_1800 = torch.constant.bool false
    %1270 = torch.aten.new_zeros %1260, %1269, %none_1797, %none_1798, %none_1799, %false_1800 : !torch.vtensor<[1,960,7,7],f32>, !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[0],f32>
    %int-1_1801 = torch.constant.int -1
    %1271 = torch.aten.unsqueeze %1261, %int-1_1801 : !torch.vtensor<[960],f32>, !torch.int -> !torch.vtensor<[960,1],f32>
    %int-1_1802 = torch.constant.int -1
    %1272 = torch.aten.unsqueeze %1271, %int-1_1802 : !torch.vtensor<[960,1],f32>, !torch.int -> !torch.vtensor<[960,1,1],f32>
    %int-1_1803 = torch.constant.int -1
    %1273 = torch.aten.unsqueeze %1266, %int-1_1803 : !torch.vtensor<[960],f32>, !torch.int -> !torch.vtensor<[960,1],f32>
    %int-1_1804 = torch.constant.int -1
    %1274 = torch.aten.unsqueeze %1273, %int-1_1804 : !torch.vtensor<[960,1],f32>, !torch.int -> !torch.vtensor<[960,1,1],f32>
    %int1_1805 = torch.constant.int 1
    %1275 = torch.aten.sub.Tensor %1260, %1272, %int1_1805 : !torch.vtensor<[1,960,7,7],f32>, !torch.vtensor<[960,1,1],f32>, !torch.int -> !torch.vtensor<[1,960,7,7],f32>
    %1276 = torch.aten.mul.Tensor %1275, %1274 : !torch.vtensor<[1,960,7,7],f32>, !torch.vtensor<[960,1,1],f32> -> !torch.vtensor<[1,960,7,7],f32>
    %int-1_1806 = torch.constant.int -1
    %1277 = torch.aten.unsqueeze %arg214, %int-1_1806 : !torch.vtensor<[960],f32>, !torch.int -> !torch.vtensor<[960,1],f32>
    %int-1_1807 = torch.constant.int -1
    %1278 = torch.aten.unsqueeze %1277, %int-1_1807 : !torch.vtensor<[960,1],f32>, !torch.int -> !torch.vtensor<[960,1,1],f32>
    %1279 = torch.aten.mul.Tensor %1276, %1278 : !torch.vtensor<[1,960,7,7],f32>, !torch.vtensor<[960,1,1],f32> -> !torch.vtensor<[1,960,7,7],f32>
    %int-1_1808 = torch.constant.int -1
    %1280 = torch.aten.unsqueeze %arg215, %int-1_1808 : !torch.vtensor<[960],f32>, !torch.int -> !torch.vtensor<[960,1],f32>
    %int-1_1809 = torch.constant.int -1
    %1281 = torch.aten.unsqueeze %1280, %int-1_1809 : !torch.vtensor<[960,1],f32>, !torch.int -> !torch.vtensor<[960,1,1],f32>
    %int1_1810 = torch.constant.int 1
    %1282 = torch.aten.add.Tensor %1279, %1281, %int1_1810 : !torch.vtensor<[1,960,7,7],f32>, !torch.vtensor<[960,1,1],f32>, !torch.int -> !torch.vtensor<[1,960,7,7],f32>
    %float0.000000e00_1811 = torch.constant.float 0.000000e+00
    %float6.000000e00_1812 = torch.constant.float 6.000000e+00
    %1283 = torch.aten.hardtanh %1282, %float0.000000e00_1811, %float6.000000e00_1812 : !torch.vtensor<[1,960,7,7],f32>, !torch.float, !torch.float -> !torch.vtensor<[1,960,7,7],f32>
    %int1_1813 = torch.constant.int 1
    %int1_1814 = torch.constant.int 1
    %int1_1815 = torch.constant.int 1
    %int1_1816 = torch.constant.int 1
    %1284 = torch.prim.ListConstruct %int1_1813, %int1_1814, %int1_1815, %int1_1816 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %float0.000000e00_1817 = torch.constant.float 0.000000e+00
    %1285 = torch.aten.constant_pad_nd %1283, %1284, %float0.000000e00_1817 : !torch.vtensor<[1,960,7,7],f32>, !torch.list<int>, !torch.float -> !torch.vtensor<[1,960,9,9],f32>
    %none_1818 = torch.constant.none
    %int1_1819 = torch.constant.int 1
    %int1_1820 = torch.constant.int 1
    %1286 = torch.prim.ListConstruct %int1_1819, %int1_1820 : (!torch.int, !torch.int) -> !torch.list<int>
    %int0_1821 = torch.constant.int 0
    %int0_1822 = torch.constant.int 0
    %1287 = torch.prim.ListConstruct %int0_1821, %int0_1822 : (!torch.int, !torch.int) -> !torch.list<int>
    %int1_1823 = torch.constant.int 1
    %int1_1824 = torch.constant.int 1
    %1288 = torch.prim.ListConstruct %int1_1823, %int1_1824 : (!torch.int, !torch.int) -> !torch.list<int>
    %false_1825 = torch.constant.bool false
    %int0_1826 = torch.constant.int 0
    %int0_1827 = torch.constant.int 0
    %1289 = torch.prim.ListConstruct %int0_1826, %int0_1827 : (!torch.int, !torch.int) -> !torch.list<int>
    %int960 = torch.constant.int 960
    %1290 = torch.aten.convolution %1285, %arg216, %none_1818, %1286, %1287, %1288, %false_1825, %1289, %int960 : !torch.vtensor<[1,960,9,9],f32>, !torch.vtensor<[960,1,3,3],f32>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[1,960,7,7],f32>
    %int6_1828 = torch.constant.int 6
    %1291 = torch.prims.convert_element_type %arg217, %int6_1828 : !torch.vtensor<[960],f32>, !torch.int -> !torch.vtensor<[960],f32>
    %int6_1829 = torch.constant.int 6
    %1292 = torch.prims.convert_element_type %arg218, %int6_1829 : !torch.vtensor<[960],f32>, !torch.int -> !torch.vtensor<[960],f32>
    %float1.000000e-03_1830 = torch.constant.float 1.000000e-03
    %int1_1831 = torch.constant.int 1
    %1293 = torch.aten.add.Scalar %1292, %float1.000000e-03_1830, %int1_1831 : !torch.vtensor<[960],f32>, !torch.float, !torch.int -> !torch.vtensor<[960],f32>
    %1294 = torch.aten.sqrt %1293 : !torch.vtensor<[960],f32> -> !torch.vtensor<[960],f32>
    %1295 = torch.aten.reciprocal %1294 : !torch.vtensor<[960],f32> -> !torch.vtensor<[960],f32>
    %int1_1832 = torch.constant.int 1
    %1296 = torch.aten.mul.Scalar %1295, %int1_1832 : !torch.vtensor<[960],f32>, !torch.int -> !torch.vtensor<[960],f32>
    %int0_1833 = torch.constant.int 0
    %1297 = torch.prim.ListConstruct %int0_1833 : (!torch.int) -> !torch.list<int>
    %none_1834 = torch.constant.none
    %none_1835 = torch.constant.none
    %none_1836 = torch.constant.none
    %false_1837 = torch.constant.bool false
    %1298 = torch.aten.new_zeros %1290, %1297, %none_1834, %none_1835, %none_1836, %false_1837 : !torch.vtensor<[1,960,7,7],f32>, !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[0],f32>
    %int0_1838 = torch.constant.int 0
    %1299 = torch.prim.ListConstruct %int0_1838 : (!torch.int) -> !torch.list<int>
    %none_1839 = torch.constant.none
    %none_1840 = torch.constant.none
    %none_1841 = torch.constant.none
    %false_1842 = torch.constant.bool false
    %1300 = torch.aten.new_zeros %1290, %1299, %none_1839, %none_1840, %none_1841, %false_1842 : !torch.vtensor<[1,960,7,7],f32>, !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[0],f32>
    %int-1_1843 = torch.constant.int -1
    %1301 = torch.aten.unsqueeze %1291, %int-1_1843 : !torch.vtensor<[960],f32>, !torch.int -> !torch.vtensor<[960,1],f32>
    %int-1_1844 = torch.constant.int -1
    %1302 = torch.aten.unsqueeze %1301, %int-1_1844 : !torch.vtensor<[960,1],f32>, !torch.int -> !torch.vtensor<[960,1,1],f32>
    %int-1_1845 = torch.constant.int -1
    %1303 = torch.aten.unsqueeze %1296, %int-1_1845 : !torch.vtensor<[960],f32>, !torch.int -> !torch.vtensor<[960,1],f32>
    %int-1_1846 = torch.constant.int -1
    %1304 = torch.aten.unsqueeze %1303, %int-1_1846 : !torch.vtensor<[960,1],f32>, !torch.int -> !torch.vtensor<[960,1,1],f32>
    %int1_1847 = torch.constant.int 1
    %1305 = torch.aten.sub.Tensor %1290, %1302, %int1_1847 : !torch.vtensor<[1,960,7,7],f32>, !torch.vtensor<[960,1,1],f32>, !torch.int -> !torch.vtensor<[1,960,7,7],f32>
    %1306 = torch.aten.mul.Tensor %1305, %1304 : !torch.vtensor<[1,960,7,7],f32>, !torch.vtensor<[960,1,1],f32> -> !torch.vtensor<[1,960,7,7],f32>
    %int-1_1848 = torch.constant.int -1
    %1307 = torch.aten.unsqueeze %arg219, %int-1_1848 : !torch.vtensor<[960],f32>, !torch.int -> !torch.vtensor<[960,1],f32>
    %int-1_1849 = torch.constant.int -1
    %1308 = torch.aten.unsqueeze %1307, %int-1_1849 : !torch.vtensor<[960,1],f32>, !torch.int -> !torch.vtensor<[960,1,1],f32>
    %1309 = torch.aten.mul.Tensor %1306, %1308 : !torch.vtensor<[1,960,7,7],f32>, !torch.vtensor<[960,1,1],f32> -> !torch.vtensor<[1,960,7,7],f32>
    %int-1_1850 = torch.constant.int -1
    %1310 = torch.aten.unsqueeze %arg220, %int-1_1850 : !torch.vtensor<[960],f32>, !torch.int -> !torch.vtensor<[960,1],f32>
    %int-1_1851 = torch.constant.int -1
    %1311 = torch.aten.unsqueeze %1310, %int-1_1851 : !torch.vtensor<[960,1],f32>, !torch.int -> !torch.vtensor<[960,1,1],f32>
    %int1_1852 = torch.constant.int 1
    %1312 = torch.aten.add.Tensor %1309, %1311, %int1_1852 : !torch.vtensor<[1,960,7,7],f32>, !torch.vtensor<[960,1,1],f32>, !torch.int -> !torch.vtensor<[1,960,7,7],f32>
    %float0.000000e00_1853 = torch.constant.float 0.000000e+00
    %float6.000000e00_1854 = torch.constant.float 6.000000e+00
    %1313 = torch.aten.hardtanh %1312, %float0.000000e00_1853, %float6.000000e00_1854 : !torch.vtensor<[1,960,7,7],f32>, !torch.float, !torch.float -> !torch.vtensor<[1,960,7,7],f32>
    %int0_1855 = torch.constant.int 0
    %int0_1856 = torch.constant.int 0
    %int0_1857 = torch.constant.int 0
    %int0_1858 = torch.constant.int 0
    %1314 = torch.prim.ListConstruct %int0_1855, %int0_1856, %int0_1857, %int0_1858 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %float0.000000e00_1859 = torch.constant.float 0.000000e+00
    %1315 = torch.aten.constant_pad_nd %1313, %1314, %float0.000000e00_1859 : !torch.vtensor<[1,960,7,7],f32>, !torch.list<int>, !torch.float -> !torch.vtensor<[1,960,7,7],f32>
    %none_1860 = torch.constant.none
    %int1_1861 = torch.constant.int 1
    %int1_1862 = torch.constant.int 1
    %1316 = torch.prim.ListConstruct %int1_1861, %int1_1862 : (!torch.int, !torch.int) -> !torch.list<int>
    %int0_1863 = torch.constant.int 0
    %int0_1864 = torch.constant.int 0
    %1317 = torch.prim.ListConstruct %int0_1863, %int0_1864 : (!torch.int, !torch.int) -> !torch.list<int>
    %int1_1865 = torch.constant.int 1
    %int1_1866 = torch.constant.int 1
    %1318 = torch.prim.ListConstruct %int1_1865, %int1_1866 : (!torch.int, !torch.int) -> !torch.list<int>
    %false_1867 = torch.constant.bool false
    %int0_1868 = torch.constant.int 0
    %int0_1869 = torch.constant.int 0
    %1319 = torch.prim.ListConstruct %int0_1868, %int0_1869 : (!torch.int, !torch.int) -> !torch.list<int>
    %int1_1870 = torch.constant.int 1
    %1320 = torch.aten.convolution %1315, %arg221, %none_1860, %1316, %1317, %1318, %false_1867, %1319, %int1_1870 : !torch.vtensor<[1,960,7,7],f32>, !torch.vtensor<[160,960,1,1],f32>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[1,160,7,7],f32>
    %int6_1871 = torch.constant.int 6
    %1321 = torch.prims.convert_element_type %arg222, %int6_1871 : !torch.vtensor<[160],f32>, !torch.int -> !torch.vtensor<[160],f32>
    %int6_1872 = torch.constant.int 6
    %1322 = torch.prims.convert_element_type %arg223, %int6_1872 : !torch.vtensor<[160],f32>, !torch.int -> !torch.vtensor<[160],f32>
    %float1.000000e-03_1873 = torch.constant.float 1.000000e-03
    %int1_1874 = torch.constant.int 1
    %1323 = torch.aten.add.Scalar %1322, %float1.000000e-03_1873, %int1_1874 : !torch.vtensor<[160],f32>, !torch.float, !torch.int -> !torch.vtensor<[160],f32>
    %1324 = torch.aten.sqrt %1323 : !torch.vtensor<[160],f32> -> !torch.vtensor<[160],f32>
    %1325 = torch.aten.reciprocal %1324 : !torch.vtensor<[160],f32> -> !torch.vtensor<[160],f32>
    %int1_1875 = torch.constant.int 1
    %1326 = torch.aten.mul.Scalar %1325, %int1_1875 : !torch.vtensor<[160],f32>, !torch.int -> !torch.vtensor<[160],f32>
    %int0_1876 = torch.constant.int 0
    %1327 = torch.prim.ListConstruct %int0_1876 : (!torch.int) -> !torch.list<int>
    %none_1877 = torch.constant.none
    %none_1878 = torch.constant.none
    %none_1879 = torch.constant.none
    %false_1880 = torch.constant.bool false
    %1328 = torch.aten.new_zeros %1320, %1327, %none_1877, %none_1878, %none_1879, %false_1880 : !torch.vtensor<[1,160,7,7],f32>, !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[0],f32>
    %int0_1881 = torch.constant.int 0
    %1329 = torch.prim.ListConstruct %int0_1881 : (!torch.int) -> !torch.list<int>
    %none_1882 = torch.constant.none
    %none_1883 = torch.constant.none
    %none_1884 = torch.constant.none
    %false_1885 = torch.constant.bool false
    %1330 = torch.aten.new_zeros %1320, %1329, %none_1882, %none_1883, %none_1884, %false_1885 : !torch.vtensor<[1,160,7,7],f32>, !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[0],f32>
    %int-1_1886 = torch.constant.int -1
    %1331 = torch.aten.unsqueeze %1321, %int-1_1886 : !torch.vtensor<[160],f32>, !torch.int -> !torch.vtensor<[160,1],f32>
    %int-1_1887 = torch.constant.int -1
    %1332 = torch.aten.unsqueeze %1331, %int-1_1887 : !torch.vtensor<[160,1],f32>, !torch.int -> !torch.vtensor<[160,1,1],f32>
    %int-1_1888 = torch.constant.int -1
    %1333 = torch.aten.unsqueeze %1326, %int-1_1888 : !torch.vtensor<[160],f32>, !torch.int -> !torch.vtensor<[160,1],f32>
    %int-1_1889 = torch.constant.int -1
    %1334 = torch.aten.unsqueeze %1333, %int-1_1889 : !torch.vtensor<[160,1],f32>, !torch.int -> !torch.vtensor<[160,1,1],f32>
    %int1_1890 = torch.constant.int 1
    %1335 = torch.aten.sub.Tensor %1320, %1332, %int1_1890 : !torch.vtensor<[1,160,7,7],f32>, !torch.vtensor<[160,1,1],f32>, !torch.int -> !torch.vtensor<[1,160,7,7],f32>
    %1336 = torch.aten.mul.Tensor %1335, %1334 : !torch.vtensor<[1,160,7,7],f32>, !torch.vtensor<[160,1,1],f32> -> !torch.vtensor<[1,160,7,7],f32>
    %int-1_1891 = torch.constant.int -1
    %1337 = torch.aten.unsqueeze %arg224, %int-1_1891 : !torch.vtensor<[160],f32>, !torch.int -> !torch.vtensor<[160,1],f32>
    %int-1_1892 = torch.constant.int -1
    %1338 = torch.aten.unsqueeze %1337, %int-1_1892 : !torch.vtensor<[160,1],f32>, !torch.int -> !torch.vtensor<[160,1,1],f32>
    %1339 = torch.aten.mul.Tensor %1336, %1338 : !torch.vtensor<[1,160,7,7],f32>, !torch.vtensor<[160,1,1],f32> -> !torch.vtensor<[1,160,7,7],f32>
    %int-1_1893 = torch.constant.int -1
    %1340 = torch.aten.unsqueeze %arg225, %int-1_1893 : !torch.vtensor<[160],f32>, !torch.int -> !torch.vtensor<[160,1],f32>
    %int-1_1894 = torch.constant.int -1
    %1341 = torch.aten.unsqueeze %1340, %int-1_1894 : !torch.vtensor<[160,1],f32>, !torch.int -> !torch.vtensor<[160,1,1],f32>
    %int1_1895 = torch.constant.int 1
    %1342 = torch.aten.add.Tensor %1339, %1341, %int1_1895 : !torch.vtensor<[1,160,7,7],f32>, !torch.vtensor<[160,1,1],f32>, !torch.int -> !torch.vtensor<[1,160,7,7],f32>
    %int1_1896 = torch.constant.int 1
    %1343 = torch.aten.add.Tensor %1253, %1342, %int1_1896 : !torch.vtensor<[1,160,7,7],f32>, !torch.vtensor<[1,160,7,7],f32>, !torch.int -> !torch.vtensor<[1,160,7,7],f32>
    %int0_1897 = torch.constant.int 0
    %int0_1898 = torch.constant.int 0
    %int0_1899 = torch.constant.int 0
    %int0_1900 = torch.constant.int 0
    %1344 = torch.prim.ListConstruct %int0_1897, %int0_1898, %int0_1899, %int0_1900 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %float0.000000e00_1901 = torch.constant.float 0.000000e+00
    %1345 = torch.aten.constant_pad_nd %1343, %1344, %float0.000000e00_1901 : !torch.vtensor<[1,160,7,7],f32>, !torch.list<int>, !torch.float -> !torch.vtensor<[1,160,7,7],f32>
    %none_1902 = torch.constant.none
    %int1_1903 = torch.constant.int 1
    %int1_1904 = torch.constant.int 1
    %1346 = torch.prim.ListConstruct %int1_1903, %int1_1904 : (!torch.int, !torch.int) -> !torch.list<int>
    %int0_1905 = torch.constant.int 0
    %int0_1906 = torch.constant.int 0
    %1347 = torch.prim.ListConstruct %int0_1905, %int0_1906 : (!torch.int, !torch.int) -> !torch.list<int>
    %int1_1907 = torch.constant.int 1
    %int1_1908 = torch.constant.int 1
    %1348 = torch.prim.ListConstruct %int1_1907, %int1_1908 : (!torch.int, !torch.int) -> !torch.list<int>
    %false_1909 = torch.constant.bool false
    %int0_1910 = torch.constant.int 0
    %int0_1911 = torch.constant.int 0
    %1349 = torch.prim.ListConstruct %int0_1910, %int0_1911 : (!torch.int, !torch.int) -> !torch.list<int>
    %int1_1912 = torch.constant.int 1
    %1350 = torch.aten.convolution %1345, %arg226, %none_1902, %1346, %1347, %1348, %false_1909, %1349, %int1_1912 : !torch.vtensor<[1,160,7,7],f32>, !torch.vtensor<[960,160,1,1],f32>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[1,960,7,7],f32>
    %int6_1913 = torch.constant.int 6
    %1351 = torch.prims.convert_element_type %arg227, %int6_1913 : !torch.vtensor<[960],f32>, !torch.int -> !torch.vtensor<[960],f32>
    %int6_1914 = torch.constant.int 6
    %1352 = torch.prims.convert_element_type %arg228, %int6_1914 : !torch.vtensor<[960],f32>, !torch.int -> !torch.vtensor<[960],f32>
    %float1.000000e-03_1915 = torch.constant.float 1.000000e-03
    %int1_1916 = torch.constant.int 1
    %1353 = torch.aten.add.Scalar %1352, %float1.000000e-03_1915, %int1_1916 : !torch.vtensor<[960],f32>, !torch.float, !torch.int -> !torch.vtensor<[960],f32>
    %1354 = torch.aten.sqrt %1353 : !torch.vtensor<[960],f32> -> !torch.vtensor<[960],f32>
    %1355 = torch.aten.reciprocal %1354 : !torch.vtensor<[960],f32> -> !torch.vtensor<[960],f32>
    %int1_1917 = torch.constant.int 1
    %1356 = torch.aten.mul.Scalar %1355, %int1_1917 : !torch.vtensor<[960],f32>, !torch.int -> !torch.vtensor<[960],f32>
    %int0_1918 = torch.constant.int 0
    %1357 = torch.prim.ListConstruct %int0_1918 : (!torch.int) -> !torch.list<int>
    %none_1919 = torch.constant.none
    %none_1920 = torch.constant.none
    %none_1921 = torch.constant.none
    %false_1922 = torch.constant.bool false
    %1358 = torch.aten.new_zeros %1350, %1357, %none_1919, %none_1920, %none_1921, %false_1922 : !torch.vtensor<[1,960,7,7],f32>, !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[0],f32>
    %int0_1923 = torch.constant.int 0
    %1359 = torch.prim.ListConstruct %int0_1923 : (!torch.int) -> !torch.list<int>
    %none_1924 = torch.constant.none
    %none_1925 = torch.constant.none
    %none_1926 = torch.constant.none
    %false_1927 = torch.constant.bool false
    %1360 = torch.aten.new_zeros %1350, %1359, %none_1924, %none_1925, %none_1926, %false_1927 : !torch.vtensor<[1,960,7,7],f32>, !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[0],f32>
    %int-1_1928 = torch.constant.int -1
    %1361 = torch.aten.unsqueeze %1351, %int-1_1928 : !torch.vtensor<[960],f32>, !torch.int -> !torch.vtensor<[960,1],f32>
    %int-1_1929 = torch.constant.int -1
    %1362 = torch.aten.unsqueeze %1361, %int-1_1929 : !torch.vtensor<[960,1],f32>, !torch.int -> !torch.vtensor<[960,1,1],f32>
    %int-1_1930 = torch.constant.int -1
    %1363 = torch.aten.unsqueeze %1356, %int-1_1930 : !torch.vtensor<[960],f32>, !torch.int -> !torch.vtensor<[960,1],f32>
    %int-1_1931 = torch.constant.int -1
    %1364 = torch.aten.unsqueeze %1363, %int-1_1931 : !torch.vtensor<[960,1],f32>, !torch.int -> !torch.vtensor<[960,1,1],f32>
    %int1_1932 = torch.constant.int 1
    %1365 = torch.aten.sub.Tensor %1350, %1362, %int1_1932 : !torch.vtensor<[1,960,7,7],f32>, !torch.vtensor<[960,1,1],f32>, !torch.int -> !torch.vtensor<[1,960,7,7],f32>
    %1366 = torch.aten.mul.Tensor %1365, %1364 : !torch.vtensor<[1,960,7,7],f32>, !torch.vtensor<[960,1,1],f32> -> !torch.vtensor<[1,960,7,7],f32>
    %int-1_1933 = torch.constant.int -1
    %1367 = torch.aten.unsqueeze %arg229, %int-1_1933 : !torch.vtensor<[960],f32>, !torch.int -> !torch.vtensor<[960,1],f32>
    %int-1_1934 = torch.constant.int -1
    %1368 = torch.aten.unsqueeze %1367, %int-1_1934 : !torch.vtensor<[960,1],f32>, !torch.int -> !torch.vtensor<[960,1,1],f32>
    %1369 = torch.aten.mul.Tensor %1366, %1368 : !torch.vtensor<[1,960,7,7],f32>, !torch.vtensor<[960,1,1],f32> -> !torch.vtensor<[1,960,7,7],f32>
    %int-1_1935 = torch.constant.int -1
    %1370 = torch.aten.unsqueeze %arg230, %int-1_1935 : !torch.vtensor<[960],f32>, !torch.int -> !torch.vtensor<[960,1],f32>
    %int-1_1936 = torch.constant.int -1
    %1371 = torch.aten.unsqueeze %1370, %int-1_1936 : !torch.vtensor<[960,1],f32>, !torch.int -> !torch.vtensor<[960,1,1],f32>
    %int1_1937 = torch.constant.int 1
    %1372 = torch.aten.add.Tensor %1369, %1371, %int1_1937 : !torch.vtensor<[1,960,7,7],f32>, !torch.vtensor<[960,1,1],f32>, !torch.int -> !torch.vtensor<[1,960,7,7],f32>
    %float0.000000e00_1938 = torch.constant.float 0.000000e+00
    %float6.000000e00_1939 = torch.constant.float 6.000000e+00
    %1373 = torch.aten.hardtanh %1372, %float0.000000e00_1938, %float6.000000e00_1939 : !torch.vtensor<[1,960,7,7],f32>, !torch.float, !torch.float -> !torch.vtensor<[1,960,7,7],f32>
    %int1_1940 = torch.constant.int 1
    %int1_1941 = torch.constant.int 1
    %int1_1942 = torch.constant.int 1
    %int1_1943 = torch.constant.int 1
    %1374 = torch.prim.ListConstruct %int1_1940, %int1_1941, %int1_1942, %int1_1943 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %float0.000000e00_1944 = torch.constant.float 0.000000e+00
    %1375 = torch.aten.constant_pad_nd %1373, %1374, %float0.000000e00_1944 : !torch.vtensor<[1,960,7,7],f32>, !torch.list<int>, !torch.float -> !torch.vtensor<[1,960,9,9],f32>
    %none_1945 = torch.constant.none
    %int1_1946 = torch.constant.int 1
    %int1_1947 = torch.constant.int 1
    %1376 = torch.prim.ListConstruct %int1_1946, %int1_1947 : (!torch.int, !torch.int) -> !torch.list<int>
    %int0_1948 = torch.constant.int 0
    %int0_1949 = torch.constant.int 0
    %1377 = torch.prim.ListConstruct %int0_1948, %int0_1949 : (!torch.int, !torch.int) -> !torch.list<int>
    %int1_1950 = torch.constant.int 1
    %int1_1951 = torch.constant.int 1
    %1378 = torch.prim.ListConstruct %int1_1950, %int1_1951 : (!torch.int, !torch.int) -> !torch.list<int>
    %false_1952 = torch.constant.bool false
    %int0_1953 = torch.constant.int 0
    %int0_1954 = torch.constant.int 0
    %1379 = torch.prim.ListConstruct %int0_1953, %int0_1954 : (!torch.int, !torch.int) -> !torch.list<int>
    %int960_1955 = torch.constant.int 960
    %1380 = torch.aten.convolution %1375, %arg231, %none_1945, %1376, %1377, %1378, %false_1952, %1379, %int960_1955 : !torch.vtensor<[1,960,9,9],f32>, !torch.vtensor<[960,1,3,3],f32>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[1,960,7,7],f32>
    %int6_1956 = torch.constant.int 6
    %1381 = torch.prims.convert_element_type %arg232, %int6_1956 : !torch.vtensor<[960],f32>, !torch.int -> !torch.vtensor<[960],f32>
    %int6_1957 = torch.constant.int 6
    %1382 = torch.prims.convert_element_type %arg233, %int6_1957 : !torch.vtensor<[960],f32>, !torch.int -> !torch.vtensor<[960],f32>
    %float1.000000e-03_1958 = torch.constant.float 1.000000e-03
    %int1_1959 = torch.constant.int 1
    %1383 = torch.aten.add.Scalar %1382, %float1.000000e-03_1958, %int1_1959 : !torch.vtensor<[960],f32>, !torch.float, !torch.int -> !torch.vtensor<[960],f32>
    %1384 = torch.aten.sqrt %1383 : !torch.vtensor<[960],f32> -> !torch.vtensor<[960],f32>
    %1385 = torch.aten.reciprocal %1384 : !torch.vtensor<[960],f32> -> !torch.vtensor<[960],f32>
    %int1_1960 = torch.constant.int 1
    %1386 = torch.aten.mul.Scalar %1385, %int1_1960 : !torch.vtensor<[960],f32>, !torch.int -> !torch.vtensor<[960],f32>
    %int0_1961 = torch.constant.int 0
    %1387 = torch.prim.ListConstruct %int0_1961 : (!torch.int) -> !torch.list<int>
    %none_1962 = torch.constant.none
    %none_1963 = torch.constant.none
    %none_1964 = torch.constant.none
    %false_1965 = torch.constant.bool false
    %1388 = torch.aten.new_zeros %1380, %1387, %none_1962, %none_1963, %none_1964, %false_1965 : !torch.vtensor<[1,960,7,7],f32>, !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[0],f32>
    %int0_1966 = torch.constant.int 0
    %1389 = torch.prim.ListConstruct %int0_1966 : (!torch.int) -> !torch.list<int>
    %none_1967 = torch.constant.none
    %none_1968 = torch.constant.none
    %none_1969 = torch.constant.none
    %false_1970 = torch.constant.bool false
    %1390 = torch.aten.new_zeros %1380, %1389, %none_1967, %none_1968, %none_1969, %false_1970 : !torch.vtensor<[1,960,7,7],f32>, !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[0],f32>
    %int-1_1971 = torch.constant.int -1
    %1391 = torch.aten.unsqueeze %1381, %int-1_1971 : !torch.vtensor<[960],f32>, !torch.int -> !torch.vtensor<[960,1],f32>
    %int-1_1972 = torch.constant.int -1
    %1392 = torch.aten.unsqueeze %1391, %int-1_1972 : !torch.vtensor<[960,1],f32>, !torch.int -> !torch.vtensor<[960,1,1],f32>
    %int-1_1973 = torch.constant.int -1
    %1393 = torch.aten.unsqueeze %1386, %int-1_1973 : !torch.vtensor<[960],f32>, !torch.int -> !torch.vtensor<[960,1],f32>
    %int-1_1974 = torch.constant.int -1
    %1394 = torch.aten.unsqueeze %1393, %int-1_1974 : !torch.vtensor<[960,1],f32>, !torch.int -> !torch.vtensor<[960,1,1],f32>
    %int1_1975 = torch.constant.int 1
    %1395 = torch.aten.sub.Tensor %1380, %1392, %int1_1975 : !torch.vtensor<[1,960,7,7],f32>, !torch.vtensor<[960,1,1],f32>, !torch.int -> !torch.vtensor<[1,960,7,7],f32>
    %1396 = torch.aten.mul.Tensor %1395, %1394 : !torch.vtensor<[1,960,7,7],f32>, !torch.vtensor<[960,1,1],f32> -> !torch.vtensor<[1,960,7,7],f32>
    %int-1_1976 = torch.constant.int -1
    %1397 = torch.aten.unsqueeze %arg234, %int-1_1976 : !torch.vtensor<[960],f32>, !torch.int -> !torch.vtensor<[960,1],f32>
    %int-1_1977 = torch.constant.int -1
    %1398 = torch.aten.unsqueeze %1397, %int-1_1977 : !torch.vtensor<[960,1],f32>, !torch.int -> !torch.vtensor<[960,1,1],f32>
    %1399 = torch.aten.mul.Tensor %1396, %1398 : !torch.vtensor<[1,960,7,7],f32>, !torch.vtensor<[960,1,1],f32> -> !torch.vtensor<[1,960,7,7],f32>
    %int-1_1978 = torch.constant.int -1
    %1400 = torch.aten.unsqueeze %arg235, %int-1_1978 : !torch.vtensor<[960],f32>, !torch.int -> !torch.vtensor<[960,1],f32>
    %int-1_1979 = torch.constant.int -1
    %1401 = torch.aten.unsqueeze %1400, %int-1_1979 : !torch.vtensor<[960,1],f32>, !torch.int -> !torch.vtensor<[960,1,1],f32>
    %int1_1980 = torch.constant.int 1
    %1402 = torch.aten.add.Tensor %1399, %1401, %int1_1980 : !torch.vtensor<[1,960,7,7],f32>, !torch.vtensor<[960,1,1],f32>, !torch.int -> !torch.vtensor<[1,960,7,7],f32>
    %float0.000000e00_1981 = torch.constant.float 0.000000e+00
    %float6.000000e00_1982 = torch.constant.float 6.000000e+00
    %1403 = torch.aten.hardtanh %1402, %float0.000000e00_1981, %float6.000000e00_1982 : !torch.vtensor<[1,960,7,7],f32>, !torch.float, !torch.float -> !torch.vtensor<[1,960,7,7],f32>
    %int0_1983 = torch.constant.int 0
    %int0_1984 = torch.constant.int 0
    %int0_1985 = torch.constant.int 0
    %int0_1986 = torch.constant.int 0
    %1404 = torch.prim.ListConstruct %int0_1983, %int0_1984, %int0_1985, %int0_1986 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %float0.000000e00_1987 = torch.constant.float 0.000000e+00
    %1405 = torch.aten.constant_pad_nd %1403, %1404, %float0.000000e00_1987 : !torch.vtensor<[1,960,7,7],f32>, !torch.list<int>, !torch.float -> !torch.vtensor<[1,960,7,7],f32>
    %none_1988 = torch.constant.none
    %int1_1989 = torch.constant.int 1
    %int1_1990 = torch.constant.int 1
    %1406 = torch.prim.ListConstruct %int1_1989, %int1_1990 : (!torch.int, !torch.int) -> !torch.list<int>
    %int0_1991 = torch.constant.int 0
    %int0_1992 = torch.constant.int 0
    %1407 = torch.prim.ListConstruct %int0_1991, %int0_1992 : (!torch.int, !torch.int) -> !torch.list<int>
    %int1_1993 = torch.constant.int 1
    %int1_1994 = torch.constant.int 1
    %1408 = torch.prim.ListConstruct %int1_1993, %int1_1994 : (!torch.int, !torch.int) -> !torch.list<int>
    %false_1995 = torch.constant.bool false
    %int0_1996 = torch.constant.int 0
    %int0_1997 = torch.constant.int 0
    %1409 = torch.prim.ListConstruct %int0_1996, %int0_1997 : (!torch.int, !torch.int) -> !torch.list<int>
    %int1_1998 = torch.constant.int 1
    %1410 = torch.aten.convolution %1405, %arg236, %none_1988, %1406, %1407, %1408, %false_1995, %1409, %int1_1998 : !torch.vtensor<[1,960,7,7],f32>, !torch.vtensor<[160,960,1,1],f32>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[1,160,7,7],f32>
    %int6_1999 = torch.constant.int 6
    %1411 = torch.prims.convert_element_type %arg237, %int6_1999 : !torch.vtensor<[160],f32>, !torch.int -> !torch.vtensor<[160],f32>
    %int6_2000 = torch.constant.int 6
    %1412 = torch.prims.convert_element_type %arg238, %int6_2000 : !torch.vtensor<[160],f32>, !torch.int -> !torch.vtensor<[160],f32>
    %float1.000000e-03_2001 = torch.constant.float 1.000000e-03
    %int1_2002 = torch.constant.int 1
    %1413 = torch.aten.add.Scalar %1412, %float1.000000e-03_2001, %int1_2002 : !torch.vtensor<[160],f32>, !torch.float, !torch.int -> !torch.vtensor<[160],f32>
    %1414 = torch.aten.sqrt %1413 : !torch.vtensor<[160],f32> -> !torch.vtensor<[160],f32>
    %1415 = torch.aten.reciprocal %1414 : !torch.vtensor<[160],f32> -> !torch.vtensor<[160],f32>
    %int1_2003 = torch.constant.int 1
    %1416 = torch.aten.mul.Scalar %1415, %int1_2003 : !torch.vtensor<[160],f32>, !torch.int -> !torch.vtensor<[160],f32>
    %int0_2004 = torch.constant.int 0
    %1417 = torch.prim.ListConstruct %int0_2004 : (!torch.int) -> !torch.list<int>
    %none_2005 = torch.constant.none
    %none_2006 = torch.constant.none
    %none_2007 = torch.constant.none
    %false_2008 = torch.constant.bool false
    %1418 = torch.aten.new_zeros %1410, %1417, %none_2005, %none_2006, %none_2007, %false_2008 : !torch.vtensor<[1,160,7,7],f32>, !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[0],f32>
    %int0_2009 = torch.constant.int 0
    %1419 = torch.prim.ListConstruct %int0_2009 : (!torch.int) -> !torch.list<int>
    %none_2010 = torch.constant.none
    %none_2011 = torch.constant.none
    %none_2012 = torch.constant.none
    %false_2013 = torch.constant.bool false
    %1420 = torch.aten.new_zeros %1410, %1419, %none_2010, %none_2011, %none_2012, %false_2013 : !torch.vtensor<[1,160,7,7],f32>, !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[0],f32>
    %int-1_2014 = torch.constant.int -1
    %1421 = torch.aten.unsqueeze %1411, %int-1_2014 : !torch.vtensor<[160],f32>, !torch.int -> !torch.vtensor<[160,1],f32>
    %int-1_2015 = torch.constant.int -1
    %1422 = torch.aten.unsqueeze %1421, %int-1_2015 : !torch.vtensor<[160,1],f32>, !torch.int -> !torch.vtensor<[160,1,1],f32>
    %int-1_2016 = torch.constant.int -1
    %1423 = torch.aten.unsqueeze %1416, %int-1_2016 : !torch.vtensor<[160],f32>, !torch.int -> !torch.vtensor<[160,1],f32>
    %int-1_2017 = torch.constant.int -1
    %1424 = torch.aten.unsqueeze %1423, %int-1_2017 : !torch.vtensor<[160,1],f32>, !torch.int -> !torch.vtensor<[160,1,1],f32>
    %int1_2018 = torch.constant.int 1
    %1425 = torch.aten.sub.Tensor %1410, %1422, %int1_2018 : !torch.vtensor<[1,160,7,7],f32>, !torch.vtensor<[160,1,1],f32>, !torch.int -> !torch.vtensor<[1,160,7,7],f32>
    %1426 = torch.aten.mul.Tensor %1425, %1424 : !torch.vtensor<[1,160,7,7],f32>, !torch.vtensor<[160,1,1],f32> -> !torch.vtensor<[1,160,7,7],f32>
    %int-1_2019 = torch.constant.int -1
    %1427 = torch.aten.unsqueeze %arg239, %int-1_2019 : !torch.vtensor<[160],f32>, !torch.int -> !torch.vtensor<[160,1],f32>
    %int-1_2020 = torch.constant.int -1
    %1428 = torch.aten.unsqueeze %1427, %int-1_2020 : !torch.vtensor<[160,1],f32>, !torch.int -> !torch.vtensor<[160,1,1],f32>
    %1429 = torch.aten.mul.Tensor %1426, %1428 : !torch.vtensor<[1,160,7,7],f32>, !torch.vtensor<[160,1,1],f32> -> !torch.vtensor<[1,160,7,7],f32>
    %int-1_2021 = torch.constant.int -1
    %1430 = torch.aten.unsqueeze %arg240, %int-1_2021 : !torch.vtensor<[160],f32>, !torch.int -> !torch.vtensor<[160,1],f32>
    %int-1_2022 = torch.constant.int -1
    %1431 = torch.aten.unsqueeze %1430, %int-1_2022 : !torch.vtensor<[160,1],f32>, !torch.int -> !torch.vtensor<[160,1,1],f32>
    %int1_2023 = torch.constant.int 1
    %1432 = torch.aten.add.Tensor %1429, %1431, %int1_2023 : !torch.vtensor<[1,160,7,7],f32>, !torch.vtensor<[160,1,1],f32>, !torch.int -> !torch.vtensor<[1,160,7,7],f32>
    %int1_2024 = torch.constant.int 1
    %1433 = torch.aten.add.Tensor %1343, %1432, %int1_2024 : !torch.vtensor<[1,160,7,7],f32>, !torch.vtensor<[1,160,7,7],f32>, !torch.int -> !torch.vtensor<[1,160,7,7],f32>
    %int0_2025 = torch.constant.int 0
    %int0_2026 = torch.constant.int 0
    %int0_2027 = torch.constant.int 0
    %int0_2028 = torch.constant.int 0
    %1434 = torch.prim.ListConstruct %int0_2025, %int0_2026, %int0_2027, %int0_2028 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %float0.000000e00_2029 = torch.constant.float 0.000000e+00
    %1435 = torch.aten.constant_pad_nd %1433, %1434, %float0.000000e00_2029 : !torch.vtensor<[1,160,7,7],f32>, !torch.list<int>, !torch.float -> !torch.vtensor<[1,160,7,7],f32>
    %none_2030 = torch.constant.none
    %int1_2031 = torch.constant.int 1
    %int1_2032 = torch.constant.int 1
    %1436 = torch.prim.ListConstruct %int1_2031, %int1_2032 : (!torch.int, !torch.int) -> !torch.list<int>
    %int0_2033 = torch.constant.int 0
    %int0_2034 = torch.constant.int 0
    %1437 = torch.prim.ListConstruct %int0_2033, %int0_2034 : (!torch.int, !torch.int) -> !torch.list<int>
    %int1_2035 = torch.constant.int 1
    %int1_2036 = torch.constant.int 1
    %1438 = torch.prim.ListConstruct %int1_2035, %int1_2036 : (!torch.int, !torch.int) -> !torch.list<int>
    %false_2037 = torch.constant.bool false
    %int0_2038 = torch.constant.int 0
    %int0_2039 = torch.constant.int 0
    %1439 = torch.prim.ListConstruct %int0_2038, %int0_2039 : (!torch.int, !torch.int) -> !torch.list<int>
    %int1_2040 = torch.constant.int 1
    %1440 = torch.aten.convolution %1435, %arg241, %none_2030, %1436, %1437, %1438, %false_2037, %1439, %int1_2040 : !torch.vtensor<[1,160,7,7],f32>, !torch.vtensor<[960,160,1,1],f32>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[1,960,7,7],f32>
    %int6_2041 = torch.constant.int 6
    %1441 = torch.prims.convert_element_type %arg242, %int6_2041 : !torch.vtensor<[960],f32>, !torch.int -> !torch.vtensor<[960],f32>
    %int6_2042 = torch.constant.int 6
    %1442 = torch.prims.convert_element_type %arg243, %int6_2042 : !torch.vtensor<[960],f32>, !torch.int -> !torch.vtensor<[960],f32>
    %float1.000000e-03_2043 = torch.constant.float 1.000000e-03
    %int1_2044 = torch.constant.int 1
    %1443 = torch.aten.add.Scalar %1442, %float1.000000e-03_2043, %int1_2044 : !torch.vtensor<[960],f32>, !torch.float, !torch.int -> !torch.vtensor<[960],f32>
    %1444 = torch.aten.sqrt %1443 : !torch.vtensor<[960],f32> -> !torch.vtensor<[960],f32>
    %1445 = torch.aten.reciprocal %1444 : !torch.vtensor<[960],f32> -> !torch.vtensor<[960],f32>
    %int1_2045 = torch.constant.int 1
    %1446 = torch.aten.mul.Scalar %1445, %int1_2045 : !torch.vtensor<[960],f32>, !torch.int -> !torch.vtensor<[960],f32>
    %int0_2046 = torch.constant.int 0
    %1447 = torch.prim.ListConstruct %int0_2046 : (!torch.int) -> !torch.list<int>
    %none_2047 = torch.constant.none
    %none_2048 = torch.constant.none
    %none_2049 = torch.constant.none
    %false_2050 = torch.constant.bool false
    %1448 = torch.aten.new_zeros %1440, %1447, %none_2047, %none_2048, %none_2049, %false_2050 : !torch.vtensor<[1,960,7,7],f32>, !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[0],f32>
    %int0_2051 = torch.constant.int 0
    %1449 = torch.prim.ListConstruct %int0_2051 : (!torch.int) -> !torch.list<int>
    %none_2052 = torch.constant.none
    %none_2053 = torch.constant.none
    %none_2054 = torch.constant.none
    %false_2055 = torch.constant.bool false
    %1450 = torch.aten.new_zeros %1440, %1449, %none_2052, %none_2053, %none_2054, %false_2055 : !torch.vtensor<[1,960,7,7],f32>, !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[0],f32>
    %int-1_2056 = torch.constant.int -1
    %1451 = torch.aten.unsqueeze %1441, %int-1_2056 : !torch.vtensor<[960],f32>, !torch.int -> !torch.vtensor<[960,1],f32>
    %int-1_2057 = torch.constant.int -1
    %1452 = torch.aten.unsqueeze %1451, %int-1_2057 : !torch.vtensor<[960,1],f32>, !torch.int -> !torch.vtensor<[960,1,1],f32>
    %int-1_2058 = torch.constant.int -1
    %1453 = torch.aten.unsqueeze %1446, %int-1_2058 : !torch.vtensor<[960],f32>, !torch.int -> !torch.vtensor<[960,1],f32>
    %int-1_2059 = torch.constant.int -1
    %1454 = torch.aten.unsqueeze %1453, %int-1_2059 : !torch.vtensor<[960,1],f32>, !torch.int -> !torch.vtensor<[960,1,1],f32>
    %int1_2060 = torch.constant.int 1
    %1455 = torch.aten.sub.Tensor %1440, %1452, %int1_2060 : !torch.vtensor<[1,960,7,7],f32>, !torch.vtensor<[960,1,1],f32>, !torch.int -> !torch.vtensor<[1,960,7,7],f32>
    %1456 = torch.aten.mul.Tensor %1455, %1454 : !torch.vtensor<[1,960,7,7],f32>, !torch.vtensor<[960,1,1],f32> -> !torch.vtensor<[1,960,7,7],f32>
    %int-1_2061 = torch.constant.int -1
    %1457 = torch.aten.unsqueeze %arg244, %int-1_2061 : !torch.vtensor<[960],f32>, !torch.int -> !torch.vtensor<[960,1],f32>
    %int-1_2062 = torch.constant.int -1
    %1458 = torch.aten.unsqueeze %1457, %int-1_2062 : !torch.vtensor<[960,1],f32>, !torch.int -> !torch.vtensor<[960,1,1],f32>
    %1459 = torch.aten.mul.Tensor %1456, %1458 : !torch.vtensor<[1,960,7,7],f32>, !torch.vtensor<[960,1,1],f32> -> !torch.vtensor<[1,960,7,7],f32>
    %int-1_2063 = torch.constant.int -1
    %1460 = torch.aten.unsqueeze %arg245, %int-1_2063 : !torch.vtensor<[960],f32>, !torch.int -> !torch.vtensor<[960,1],f32>
    %int-1_2064 = torch.constant.int -1
    %1461 = torch.aten.unsqueeze %1460, %int-1_2064 : !torch.vtensor<[960,1],f32>, !torch.int -> !torch.vtensor<[960,1,1],f32>
    %int1_2065 = torch.constant.int 1
    %1462 = torch.aten.add.Tensor %1459, %1461, %int1_2065 : !torch.vtensor<[1,960,7,7],f32>, !torch.vtensor<[960,1,1],f32>, !torch.int -> !torch.vtensor<[1,960,7,7],f32>
    %float0.000000e00_2066 = torch.constant.float 0.000000e+00
    %float6.000000e00_2067 = torch.constant.float 6.000000e+00
    %1463 = torch.aten.hardtanh %1462, %float0.000000e00_2066, %float6.000000e00_2067 : !torch.vtensor<[1,960,7,7],f32>, !torch.float, !torch.float -> !torch.vtensor<[1,960,7,7],f32>
    %int1_2068 = torch.constant.int 1
    %int1_2069 = torch.constant.int 1
    %int1_2070 = torch.constant.int 1
    %int1_2071 = torch.constant.int 1
    %1464 = torch.prim.ListConstruct %int1_2068, %int1_2069, %int1_2070, %int1_2071 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %float0.000000e00_2072 = torch.constant.float 0.000000e+00
    %1465 = torch.aten.constant_pad_nd %1463, %1464, %float0.000000e00_2072 : !torch.vtensor<[1,960,7,7],f32>, !torch.list<int>, !torch.float -> !torch.vtensor<[1,960,9,9],f32>
    %none_2073 = torch.constant.none
    %int1_2074 = torch.constant.int 1
    %int1_2075 = torch.constant.int 1
    %1466 = torch.prim.ListConstruct %int1_2074, %int1_2075 : (!torch.int, !torch.int) -> !torch.list<int>
    %int0_2076 = torch.constant.int 0
    %int0_2077 = torch.constant.int 0
    %1467 = torch.prim.ListConstruct %int0_2076, %int0_2077 : (!torch.int, !torch.int) -> !torch.list<int>
    %int1_2078 = torch.constant.int 1
    %int1_2079 = torch.constant.int 1
    %1468 = torch.prim.ListConstruct %int1_2078, %int1_2079 : (!torch.int, !torch.int) -> !torch.list<int>
    %false_2080 = torch.constant.bool false
    %int0_2081 = torch.constant.int 0
    %int0_2082 = torch.constant.int 0
    %1469 = torch.prim.ListConstruct %int0_2081, %int0_2082 : (!torch.int, !torch.int) -> !torch.list<int>
    %int960_2083 = torch.constant.int 960
    %1470 = torch.aten.convolution %1465, %arg246, %none_2073, %1466, %1467, %1468, %false_2080, %1469, %int960_2083 : !torch.vtensor<[1,960,9,9],f32>, !torch.vtensor<[960,1,3,3],f32>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[1,960,7,7],f32>
    %int6_2084 = torch.constant.int 6
    %1471 = torch.prims.convert_element_type %arg247, %int6_2084 : !torch.vtensor<[960],f32>, !torch.int -> !torch.vtensor<[960],f32>
    %int6_2085 = torch.constant.int 6
    %1472 = torch.prims.convert_element_type %arg248, %int6_2085 : !torch.vtensor<[960],f32>, !torch.int -> !torch.vtensor<[960],f32>
    %float1.000000e-03_2086 = torch.constant.float 1.000000e-03
    %int1_2087 = torch.constant.int 1
    %1473 = torch.aten.add.Scalar %1472, %float1.000000e-03_2086, %int1_2087 : !torch.vtensor<[960],f32>, !torch.float, !torch.int -> !torch.vtensor<[960],f32>
    %1474 = torch.aten.sqrt %1473 : !torch.vtensor<[960],f32> -> !torch.vtensor<[960],f32>
    %1475 = torch.aten.reciprocal %1474 : !torch.vtensor<[960],f32> -> !torch.vtensor<[960],f32>
    %int1_2088 = torch.constant.int 1
    %1476 = torch.aten.mul.Scalar %1475, %int1_2088 : !torch.vtensor<[960],f32>, !torch.int -> !torch.vtensor<[960],f32>
    %int0_2089 = torch.constant.int 0
    %1477 = torch.prim.ListConstruct %int0_2089 : (!torch.int) -> !torch.list<int>
    %none_2090 = torch.constant.none
    %none_2091 = torch.constant.none
    %none_2092 = torch.constant.none
    %false_2093 = torch.constant.bool false
    %1478 = torch.aten.new_zeros %1470, %1477, %none_2090, %none_2091, %none_2092, %false_2093 : !torch.vtensor<[1,960,7,7],f32>, !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[0],f32>
    %int0_2094 = torch.constant.int 0
    %1479 = torch.prim.ListConstruct %int0_2094 : (!torch.int) -> !torch.list<int>
    %none_2095 = torch.constant.none
    %none_2096 = torch.constant.none
    %none_2097 = torch.constant.none
    %false_2098 = torch.constant.bool false
    %1480 = torch.aten.new_zeros %1470, %1479, %none_2095, %none_2096, %none_2097, %false_2098 : !torch.vtensor<[1,960,7,7],f32>, !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[0],f32>
    %int-1_2099 = torch.constant.int -1
    %1481 = torch.aten.unsqueeze %1471, %int-1_2099 : !torch.vtensor<[960],f32>, !torch.int -> !torch.vtensor<[960,1],f32>
    %int-1_2100 = torch.constant.int -1
    %1482 = torch.aten.unsqueeze %1481, %int-1_2100 : !torch.vtensor<[960,1],f32>, !torch.int -> !torch.vtensor<[960,1,1],f32>
    %int-1_2101 = torch.constant.int -1
    %1483 = torch.aten.unsqueeze %1476, %int-1_2101 : !torch.vtensor<[960],f32>, !torch.int -> !torch.vtensor<[960,1],f32>
    %int-1_2102 = torch.constant.int -1
    %1484 = torch.aten.unsqueeze %1483, %int-1_2102 : !torch.vtensor<[960,1],f32>, !torch.int -> !torch.vtensor<[960,1,1],f32>
    %int1_2103 = torch.constant.int 1
    %1485 = torch.aten.sub.Tensor %1470, %1482, %int1_2103 : !torch.vtensor<[1,960,7,7],f32>, !torch.vtensor<[960,1,1],f32>, !torch.int -> !torch.vtensor<[1,960,7,7],f32>
    %1486 = torch.aten.mul.Tensor %1485, %1484 : !torch.vtensor<[1,960,7,7],f32>, !torch.vtensor<[960,1,1],f32> -> !torch.vtensor<[1,960,7,7],f32>
    %int-1_2104 = torch.constant.int -1
    %1487 = torch.aten.unsqueeze %arg249, %int-1_2104 : !torch.vtensor<[960],f32>, !torch.int -> !torch.vtensor<[960,1],f32>
    %int-1_2105 = torch.constant.int -1
    %1488 = torch.aten.unsqueeze %1487, %int-1_2105 : !torch.vtensor<[960,1],f32>, !torch.int -> !torch.vtensor<[960,1,1],f32>
    %1489 = torch.aten.mul.Tensor %1486, %1488 : !torch.vtensor<[1,960,7,7],f32>, !torch.vtensor<[960,1,1],f32> -> !torch.vtensor<[1,960,7,7],f32>
    %int-1_2106 = torch.constant.int -1
    %1490 = torch.aten.unsqueeze %arg250, %int-1_2106 : !torch.vtensor<[960],f32>, !torch.int -> !torch.vtensor<[960,1],f32>
    %int-1_2107 = torch.constant.int -1
    %1491 = torch.aten.unsqueeze %1490, %int-1_2107 : !torch.vtensor<[960,1],f32>, !torch.int -> !torch.vtensor<[960,1,1],f32>
    %int1_2108 = torch.constant.int 1
    %1492 = torch.aten.add.Tensor %1489, %1491, %int1_2108 : !torch.vtensor<[1,960,7,7],f32>, !torch.vtensor<[960,1,1],f32>, !torch.int -> !torch.vtensor<[1,960,7,7],f32>
    %float0.000000e00_2109 = torch.constant.float 0.000000e+00
    %float6.000000e00_2110 = torch.constant.float 6.000000e+00
    %1493 = torch.aten.hardtanh %1492, %float0.000000e00_2109, %float6.000000e00_2110 : !torch.vtensor<[1,960,7,7],f32>, !torch.float, !torch.float -> !torch.vtensor<[1,960,7,7],f32>
    %int0_2111 = torch.constant.int 0
    %int0_2112 = torch.constant.int 0
    %int0_2113 = torch.constant.int 0
    %int0_2114 = torch.constant.int 0
    %1494 = torch.prim.ListConstruct %int0_2111, %int0_2112, %int0_2113, %int0_2114 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %float0.000000e00_2115 = torch.constant.float 0.000000e+00
    %1495 = torch.aten.constant_pad_nd %1493, %1494, %float0.000000e00_2115 : !torch.vtensor<[1,960,7,7],f32>, !torch.list<int>, !torch.float -> !torch.vtensor<[1,960,7,7],f32>
    %none_2116 = torch.constant.none
    %int1_2117 = torch.constant.int 1
    %int1_2118 = torch.constant.int 1
    %1496 = torch.prim.ListConstruct %int1_2117, %int1_2118 : (!torch.int, !torch.int) -> !torch.list<int>
    %int0_2119 = torch.constant.int 0
    %int0_2120 = torch.constant.int 0
    %1497 = torch.prim.ListConstruct %int0_2119, %int0_2120 : (!torch.int, !torch.int) -> !torch.list<int>
    %int1_2121 = torch.constant.int 1
    %int1_2122 = torch.constant.int 1
    %1498 = torch.prim.ListConstruct %int1_2121, %int1_2122 : (!torch.int, !torch.int) -> !torch.list<int>
    %false_2123 = torch.constant.bool false
    %int0_2124 = torch.constant.int 0
    %int0_2125 = torch.constant.int 0
    %1499 = torch.prim.ListConstruct %int0_2124, %int0_2125 : (!torch.int, !torch.int) -> !torch.list<int>
    %int1_2126 = torch.constant.int 1
    %1500 = torch.aten.convolution %1495, %arg251, %none_2116, %1496, %1497, %1498, %false_2123, %1499, %int1_2126 : !torch.vtensor<[1,960,7,7],f32>, !torch.vtensor<[320,960,1,1],f32>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[1,320,7,7],f32>
    %int6_2127 = torch.constant.int 6
    %1501 = torch.prims.convert_element_type %arg252, %int6_2127 : !torch.vtensor<[320],f32>, !torch.int -> !torch.vtensor<[320],f32>
    %int6_2128 = torch.constant.int 6
    %1502 = torch.prims.convert_element_type %arg253, %int6_2128 : !torch.vtensor<[320],f32>, !torch.int -> !torch.vtensor<[320],f32>
    %float1.000000e-03_2129 = torch.constant.float 1.000000e-03
    %int1_2130 = torch.constant.int 1
    %1503 = torch.aten.add.Scalar %1502, %float1.000000e-03_2129, %int1_2130 : !torch.vtensor<[320],f32>, !torch.float, !torch.int -> !torch.vtensor<[320],f32>
    %1504 = torch.aten.sqrt %1503 : !torch.vtensor<[320],f32> -> !torch.vtensor<[320],f32>
    %1505 = torch.aten.reciprocal %1504 : !torch.vtensor<[320],f32> -> !torch.vtensor<[320],f32>
    %int1_2131 = torch.constant.int 1
    %1506 = torch.aten.mul.Scalar %1505, %int1_2131 : !torch.vtensor<[320],f32>, !torch.int -> !torch.vtensor<[320],f32>
    %int0_2132 = torch.constant.int 0
    %1507 = torch.prim.ListConstruct %int0_2132 : (!torch.int) -> !torch.list<int>
    %none_2133 = torch.constant.none
    %none_2134 = torch.constant.none
    %none_2135 = torch.constant.none
    %false_2136 = torch.constant.bool false
    %1508 = torch.aten.new_zeros %1500, %1507, %none_2133, %none_2134, %none_2135, %false_2136 : !torch.vtensor<[1,320,7,7],f32>, !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[0],f32>
    %int0_2137 = torch.constant.int 0
    %1509 = torch.prim.ListConstruct %int0_2137 : (!torch.int) -> !torch.list<int>
    %none_2138 = torch.constant.none
    %none_2139 = torch.constant.none
    %none_2140 = torch.constant.none
    %false_2141 = torch.constant.bool false
    %1510 = torch.aten.new_zeros %1500, %1509, %none_2138, %none_2139, %none_2140, %false_2141 : !torch.vtensor<[1,320,7,7],f32>, !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[0],f32>
    %int-1_2142 = torch.constant.int -1
    %1511 = torch.aten.unsqueeze %1501, %int-1_2142 : !torch.vtensor<[320],f32>, !torch.int -> !torch.vtensor<[320,1],f32>
    %int-1_2143 = torch.constant.int -1
    %1512 = torch.aten.unsqueeze %1511, %int-1_2143 : !torch.vtensor<[320,1],f32>, !torch.int -> !torch.vtensor<[320,1,1],f32>
    %int-1_2144 = torch.constant.int -1
    %1513 = torch.aten.unsqueeze %1506, %int-1_2144 : !torch.vtensor<[320],f32>, !torch.int -> !torch.vtensor<[320,1],f32>
    %int-1_2145 = torch.constant.int -1
    %1514 = torch.aten.unsqueeze %1513, %int-1_2145 : !torch.vtensor<[320,1],f32>, !torch.int -> !torch.vtensor<[320,1,1],f32>
    %int1_2146 = torch.constant.int 1
    %1515 = torch.aten.sub.Tensor %1500, %1512, %int1_2146 : !torch.vtensor<[1,320,7,7],f32>, !torch.vtensor<[320,1,1],f32>, !torch.int -> !torch.vtensor<[1,320,7,7],f32>
    %1516 = torch.aten.mul.Tensor %1515, %1514 : !torch.vtensor<[1,320,7,7],f32>, !torch.vtensor<[320,1,1],f32> -> !torch.vtensor<[1,320,7,7],f32>
    %int-1_2147 = torch.constant.int -1
    %1517 = torch.aten.unsqueeze %arg254, %int-1_2147 : !torch.vtensor<[320],f32>, !torch.int -> !torch.vtensor<[320,1],f32>
    %int-1_2148 = torch.constant.int -1
    %1518 = torch.aten.unsqueeze %1517, %int-1_2148 : !torch.vtensor<[320,1],f32>, !torch.int -> !torch.vtensor<[320,1,1],f32>
    %1519 = torch.aten.mul.Tensor %1516, %1518 : !torch.vtensor<[1,320,7,7],f32>, !torch.vtensor<[320,1,1],f32> -> !torch.vtensor<[1,320,7,7],f32>
    %int-1_2149 = torch.constant.int -1
    %1520 = torch.aten.unsqueeze %arg255, %int-1_2149 : !torch.vtensor<[320],f32>, !torch.int -> !torch.vtensor<[320,1],f32>
    %int-1_2150 = torch.constant.int -1
    %1521 = torch.aten.unsqueeze %1520, %int-1_2150 : !torch.vtensor<[320,1],f32>, !torch.int -> !torch.vtensor<[320,1,1],f32>
    %int1_2151 = torch.constant.int 1
    %1522 = torch.aten.add.Tensor %1519, %1521, %int1_2151 : !torch.vtensor<[1,320,7,7],f32>, !torch.vtensor<[320,1,1],f32>, !torch.int -> !torch.vtensor<[1,320,7,7],f32>
    %int0_2152 = torch.constant.int 0
    %int0_2153 = torch.constant.int 0
    %int0_2154 = torch.constant.int 0
    %int0_2155 = torch.constant.int 0
    %1523 = torch.prim.ListConstruct %int0_2152, %int0_2153, %int0_2154, %int0_2155 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %float0.000000e00_2156 = torch.constant.float 0.000000e+00
    %1524 = torch.aten.constant_pad_nd %1522, %1523, %float0.000000e00_2156 : !torch.vtensor<[1,320,7,7],f32>, !torch.list<int>, !torch.float -> !torch.vtensor<[1,320,7,7],f32>
    %none_2157 = torch.constant.none
    %int1_2158 = torch.constant.int 1
    %int1_2159 = torch.constant.int 1
    %1525 = torch.prim.ListConstruct %int1_2158, %int1_2159 : (!torch.int, !torch.int) -> !torch.list<int>
    %int0_2160 = torch.constant.int 0
    %int0_2161 = torch.constant.int 0
    %1526 = torch.prim.ListConstruct %int0_2160, %int0_2161 : (!torch.int, !torch.int) -> !torch.list<int>
    %int1_2162 = torch.constant.int 1
    %int1_2163 = torch.constant.int 1
    %1527 = torch.prim.ListConstruct %int1_2162, %int1_2163 : (!torch.int, !torch.int) -> !torch.list<int>
    %false_2164 = torch.constant.bool false
    %int0_2165 = torch.constant.int 0
    %int0_2166 = torch.constant.int 0
    %1528 = torch.prim.ListConstruct %int0_2165, %int0_2166 : (!torch.int, !torch.int) -> !torch.list<int>
    %int1_2167 = torch.constant.int 1
    %1529 = torch.aten.convolution %1524, %arg256, %none_2157, %1525, %1526, %1527, %false_2164, %1528, %int1_2167 : !torch.vtensor<[1,320,7,7],f32>, !torch.vtensor<[1280,320,1,1],f32>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[1,1280,7,7],f32>
    %int6_2168 = torch.constant.int 6
    %1530 = torch.prims.convert_element_type %arg257, %int6_2168 : !torch.vtensor<[1280],f32>, !torch.int -> !torch.vtensor<[1280],f32>
    %int6_2169 = torch.constant.int 6
    %1531 = torch.prims.convert_element_type %arg258, %int6_2169 : !torch.vtensor<[1280],f32>, !torch.int -> !torch.vtensor<[1280],f32>
    %float1.000000e-03_2170 = torch.constant.float 1.000000e-03
    %int1_2171 = torch.constant.int 1
    %1532 = torch.aten.add.Scalar %1531, %float1.000000e-03_2170, %int1_2171 : !torch.vtensor<[1280],f32>, !torch.float, !torch.int -> !torch.vtensor<[1280],f32>
    %1533 = torch.aten.sqrt %1532 : !torch.vtensor<[1280],f32> -> !torch.vtensor<[1280],f32>
    %1534 = torch.aten.reciprocal %1533 : !torch.vtensor<[1280],f32> -> !torch.vtensor<[1280],f32>
    %int1_2172 = torch.constant.int 1
    %1535 = torch.aten.mul.Scalar %1534, %int1_2172 : !torch.vtensor<[1280],f32>, !torch.int -> !torch.vtensor<[1280],f32>
    %int0_2173 = torch.constant.int 0
    %1536 = torch.prim.ListConstruct %int0_2173 : (!torch.int) -> !torch.list<int>
    %none_2174 = torch.constant.none
    %none_2175 = torch.constant.none
    %none_2176 = torch.constant.none
    %false_2177 = torch.constant.bool false
    %1537 = torch.aten.new_zeros %1529, %1536, %none_2174, %none_2175, %none_2176, %false_2177 : !torch.vtensor<[1,1280,7,7],f32>, !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[0],f32>
    %int0_2178 = torch.constant.int 0
    %1538 = torch.prim.ListConstruct %int0_2178 : (!torch.int) -> !torch.list<int>
    %none_2179 = torch.constant.none
    %none_2180 = torch.constant.none
    %none_2181 = torch.constant.none
    %false_2182 = torch.constant.bool false
    %1539 = torch.aten.new_zeros %1529, %1538, %none_2179, %none_2180, %none_2181, %false_2182 : !torch.vtensor<[1,1280,7,7],f32>, !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[0],f32>
    %int-1_2183 = torch.constant.int -1
    %1540 = torch.aten.unsqueeze %1530, %int-1_2183 : !torch.vtensor<[1280],f32>, !torch.int -> !torch.vtensor<[1280,1],f32>
    %int-1_2184 = torch.constant.int -1
    %1541 = torch.aten.unsqueeze %1540, %int-1_2184 : !torch.vtensor<[1280,1],f32>, !torch.int -> !torch.vtensor<[1280,1,1],f32>
    %int-1_2185 = torch.constant.int -1
    %1542 = torch.aten.unsqueeze %1535, %int-1_2185 : !torch.vtensor<[1280],f32>, !torch.int -> !torch.vtensor<[1280,1],f32>
    %int-1_2186 = torch.constant.int -1
    %1543 = torch.aten.unsqueeze %1542, %int-1_2186 : !torch.vtensor<[1280,1],f32>, !torch.int -> !torch.vtensor<[1280,1,1],f32>
    %int1_2187 = torch.constant.int 1
    %1544 = torch.aten.sub.Tensor %1529, %1541, %int1_2187 : !torch.vtensor<[1,1280,7,7],f32>, !torch.vtensor<[1280,1,1],f32>, !torch.int -> !torch.vtensor<[1,1280,7,7],f32>
    %1545 = torch.aten.mul.Tensor %1544, %1543 : !torch.vtensor<[1,1280,7,7],f32>, !torch.vtensor<[1280,1,1],f32> -> !torch.vtensor<[1,1280,7,7],f32>
    %int-1_2188 = torch.constant.int -1
    %1546 = torch.aten.unsqueeze %arg259, %int-1_2188 : !torch.vtensor<[1280],f32>, !torch.int -> !torch.vtensor<[1280,1],f32>
    %int-1_2189 = torch.constant.int -1
    %1547 = torch.aten.unsqueeze %1546, %int-1_2189 : !torch.vtensor<[1280,1],f32>, !torch.int -> !torch.vtensor<[1280,1,1],f32>
    %1548 = torch.aten.mul.Tensor %1545, %1547 : !torch.vtensor<[1,1280,7,7],f32>, !torch.vtensor<[1280,1,1],f32> -> !torch.vtensor<[1,1280,7,7],f32>
    %int-1_2190 = torch.constant.int -1
    %1549 = torch.aten.unsqueeze %arg260, %int-1_2190 : !torch.vtensor<[1280],f32>, !torch.int -> !torch.vtensor<[1280,1],f32>
    %int-1_2191 = torch.constant.int -1
    %1550 = torch.aten.unsqueeze %1549, %int-1_2191 : !torch.vtensor<[1280,1],f32>, !torch.int -> !torch.vtensor<[1280,1,1],f32>
    %int1_2192 = torch.constant.int 1
    %1551 = torch.aten.add.Tensor %1548, %1550, %int1_2192 : !torch.vtensor<[1,1280,7,7],f32>, !torch.vtensor<[1280,1,1],f32>, !torch.int -> !torch.vtensor<[1,1280,7,7],f32>
    %float0.000000e00_2193 = torch.constant.float 0.000000e+00
    %float6.000000e00_2194 = torch.constant.float 6.000000e+00
    %1552 = torch.aten.hardtanh %1551, %float0.000000e00_2193, %float6.000000e00_2194 : !torch.vtensor<[1,1280,7,7],f32>, !torch.float, !torch.float -> !torch.vtensor<[1,1280,7,7],f32>
    %int-1_2195 = torch.constant.int -1
    %int-2 = torch.constant.int -2
    %1553 = torch.prim.ListConstruct %int-1_2195, %int-2 : (!torch.int, !torch.int) -> !torch.list<int>
    %true = torch.constant.bool true
    %none_2196 = torch.constant.none
    %1554 = torch.aten.mean.dim %1552, %1553, %true, %none_2196 : !torch.vtensor<[1,1280,7,7],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[1,1280,1,1],f32>
    %int1_2197 = torch.constant.int 1
    %int1280 = torch.constant.int 1280
    %1555 = torch.prim.ListConstruct %int1_2197, %int1280 : (!torch.int, !torch.int) -> !torch.list<int>
    %1556 = torch.aten.view %1554, %1555 : !torch.vtensor<[1,1280,1,1],f32>, !torch.list<int> -> !torch.vtensor<[1,1280],f32>
    %int0_2198 = torch.constant.int 0
    %int1_2199 = torch.constant.int 1
    %1557 = torch.aten.transpose.int %arg261, %int0_2198, %int1_2199 : !torch.vtensor<[1001,1280],f32>, !torch.int, !torch.int -> !torch.vtensor<[1280,1001],f32>
    %1558 = torch.aten.mm %1556, %1557 : !torch.vtensor<[1,1280],f32>, !torch.vtensor<[1280,1001],f32> -> !torch.vtensor<[1,1001],f32>
    %int1_2200 = torch.constant.int 1
    %1559 = torch.aten.mul.Scalar %1558, %int1_2200 : !torch.vtensor<[1,1001],f32>, !torch.int -> !torch.vtensor<[1,1001],f32>
    %int1_2201 = torch.constant.int 1
    %1560 = torch.aten.mul.Scalar %arg262, %int1_2201 : !torch.vtensor<[1001],f32>, !torch.int -> !torch.vtensor<[1001],f32>
    %int1_2202 = torch.constant.int 1
    %1561 = torch.aten.add.Tensor %1559, %1560, %int1_2202 : !torch.vtensor<[1,1001],f32>, !torch.vtensor<[1001],f32>, !torch.int -> !torch.vtensor<[1,1001],f32>
    return %1561, %arg1, %arg2, %arg3, %arg4, %arg6, %arg7, %arg8, %arg9, %arg11, %arg12, %arg13, %arg14, %arg16, %arg17, %arg18, %arg19, %arg21, %arg22, %arg23, %arg24, %arg26, %arg27, %arg28, %arg29, %arg31, %arg32, %arg33, %arg34, %arg36, %arg37, %arg38, %arg39, %arg41, %arg42, %arg43, %arg44, %arg46, %arg47, %arg48, %arg49, %arg51, %arg52, %arg53, %arg54, %arg56, %arg57, %arg58, %arg59, %arg61, %arg62, %arg63, %arg64, %arg66, %arg67, %arg68, %arg69, %arg71, %arg72, %arg73, %arg74, %arg76, %arg77, %arg78, %arg79, %arg81, %arg82, %arg83, %arg84, %arg86, %arg87, %arg88, %arg89, %arg91, %arg92, %arg93, %arg94, %arg96, %arg97, %arg98, %arg99, %arg101, %arg102, %arg103, %arg104, %arg106, %arg107, %arg108, %arg109, %arg111, %arg112, %arg113, %arg114, %arg116, %arg117, %arg118, %arg119, %arg121, %arg122, %arg123, %arg124, %arg126, %arg127, %arg128, %arg129, %arg131, %arg132, %arg133, %arg134, %arg136, %arg137, %arg138, %arg139, %arg141, %arg142, %arg143, %arg144, %arg146, %arg147, %arg148, %arg149, %arg151, %arg152, %arg153, %arg154, %arg156, %arg157, %arg158, %arg159, %arg161, %arg162, %arg163, %arg164, %arg166, %arg167, %arg168, %arg169, %arg171, %arg172, %arg173, %arg174, %arg176, %arg177, %arg178, %arg179, %arg181, %arg182, %arg183, %arg184, %arg186, %arg187, %arg188, %arg189, %arg191, %arg192, %arg193, %arg194, %arg196, %arg197, %arg198, %arg199, %arg201, %arg202, %arg203, %arg204, %arg206, %arg207, %arg208, %arg209, %arg211, %arg212, %arg213, %arg214, %arg216, %arg217, %arg218, %arg219, %arg221, %arg222, %arg223, %arg224, %arg226, %arg227, %arg228, %arg229, %arg231, %arg232, %arg233, %arg234, %arg236, %arg237, %arg238, %arg239, %arg241, %arg242, %arg243, %arg244, %arg246, %arg247, %arg248, %arg249, %arg251, %arg252, %arg253, %arg254, %arg256, %arg257, %arg258, %arg259, %1, %6, %28, %14, %16, %31, %36, %58, %44, %46, %61, %66, %74, %76, %90, %95, %117, %103, %105, %120, %125, %147, %133, %135, %150, %155, %163, %165, %179, %184, %206, %192, %194, %209, %214, %236, %222, %224, %239, %244, %252, %254, %269, %274, %296, %282, %284, %299, %304, %326, %312, %314, %329, %334, %342, %344, %358, %363, %385, %371, %373, %388, %393, %415, %401, %403, %418, %423, %431, %433, %448, %453, %475, %461, %463, %478, %483, %505, %491, %493, %508, %513, %521, %523, %538, %543, %565, %551, %553, %568, %573, %595, %581, %583, %598, %603, %611, %613, %627, %632, %654, %640, %642, %657, %662, %684, %670, %672, %687, %692, %700, %702, %717, %722, %744, %730, %732, %747, %752, %774, %760, %762, %777, %782, %790, %792, %807, %812, %834, %820, %822, %837, %842, %864, %850, %852, %867, %872, %880, %882, %897, %902, %924, %910, %912, %927, %932, %954, %940, %942, %957, %962, %970, %972, %986, %991, %1013, %999, %1001, %1016, %1021, %1043, %1029, %1031, %1046, %1051, %1059, %1061, %1076, %1081, %1103, %1089, %1091, %1106, %1111, %1133, %1119, %1121, %1136, %1141, %1149, %1151, %1166, %1171, %1193, %1179, %1181, %1196, %1201, %1223, %1209, %1211, %1226, %1231, %1239, %1241, %1255, %1260, %1282, %1268, %1270, %1285, %1290, %1312, %1298, %1300, %1315, %1320, %1328, %1330, %1345, %1350, %1372, %1358, %1360, %1375, %1380, %1402, %1388, %1390, %1405, %1410, %1418, %1420, %1435, %1440, %1462, %1448, %1450, %1465, %1470, %1492, %1478, %1480, %1495, %1500, %1508, %1510, %1524, %1529, %1551, %1537, %1539, %1556, %1557 : !torch.vtensor<[1,1001],f32>, !torch.vtensor<[32,3,3,3],f32>, !torch.vtensor<[32],f32>, !torch.vtensor<[32],f32>, !torch.vtensor<[32],f32>, !torch.vtensor<[32,1,3,3],f32>, !torch.vtensor<[32],f32>, !torch.vtensor<[32],f32>, !torch.vtensor<[32],f32>, !torch.vtensor<[16,32,1,1],f32>, !torch.vtensor<[16],f32>, !torch.vtensor<[16],f32>, !torch.vtensor<[16],f32>, !torch.vtensor<[96,16,1,1],f32>, !torch.vtensor<[96],f32>, !torch.vtensor<[96],f32>, !torch.vtensor<[96],f32>, !torch.vtensor<[96,1,3,3],f32>, !torch.vtensor<[96],f32>, !torch.vtensor<[96],f32>, !torch.vtensor<[96],f32>, !torch.vtensor<[24,96,1,1],f32>, !torch.vtensor<[24],f32>, !torch.vtensor<[24],f32>, !torch.vtensor<[24],f32>, !torch.vtensor<[144,24,1,1],f32>, !torch.vtensor<[144],f32>, !torch.vtensor<[144],f32>, !torch.vtensor<[144],f32>, !torch.vtensor<[144,1,3,3],f32>, !torch.vtensor<[144],f32>, !torch.vtensor<[144],f32>, !torch.vtensor<[144],f32>, !torch.vtensor<[24,144,1,1],f32>, !torch.vtensor<[24],f32>, !torch.vtensor<[24],f32>, !torch.vtensor<[24],f32>, !torch.vtensor<[144,24,1,1],f32>, !torch.vtensor<[144],f32>, !torch.vtensor<[144],f32>, !torch.vtensor<[144],f32>, !torch.vtensor<[144,1,3,3],f32>, !torch.vtensor<[144],f32>, !torch.vtensor<[144],f32>, !torch.vtensor<[144],f32>, !torch.vtensor<[32,144,1,1],f32>, !torch.vtensor<[32],f32>, !torch.vtensor<[32],f32>, !torch.vtensor<[32],f32>, !torch.vtensor<[192,32,1,1],f32>, !torch.vtensor<[192],f32>, !torch.vtensor<[192],f32>, !torch.vtensor<[192],f32>, !torch.vtensor<[192,1,3,3],f32>, !torch.vtensor<[192],f32>, !torch.vtensor<[192],f32>, !torch.vtensor<[192],f32>, !torch.vtensor<[32,192,1,1],f32>, !torch.vtensor<[32],f32>, !torch.vtensor<[32],f32>, !torch.vtensor<[32],f32>, !torch.vtensor<[192,32,1,1],f32>, !torch.vtensor<[192],f32>, !torch.vtensor<[192],f32>, !torch.vtensor<[192],f32>, !torch.vtensor<[192,1,3,3],f32>, !torch.vtensor<[192],f32>, !torch.vtensor<[192],f32>, !torch.vtensor<[192],f32>, !torch.vtensor<[32,192,1,1],f32>, !torch.vtensor<[32],f32>, !torch.vtensor<[32],f32>, !torch.vtensor<[32],f32>, !torch.vtensor<[192,32,1,1],f32>, !torch.vtensor<[192],f32>, !torch.vtensor<[192],f32>, !torch.vtensor<[192],f32>, !torch.vtensor<[192,1,3,3],f32>, !torch.vtensor<[192],f32>, !torch.vtensor<[192],f32>, !torch.vtensor<[192],f32>, !torch.vtensor<[64,192,1,1],f32>, !torch.vtensor<[64],f32>, !torch.vtensor<[64],f32>, !torch.vtensor<[64],f32>, !torch.vtensor<[384,64,1,1],f32>, !torch.vtensor<[384],f32>, !torch.vtensor<[384],f32>, !torch.vtensor<[384],f32>, !torch.vtensor<[384,1,3,3],f32>, !torch.vtensor<[384],f32>, !torch.vtensor<[384],f32>, !torch.vtensor<[384],f32>, !torch.vtensor<[64,384,1,1],f32>, !torch.vtensor<[64],f32>, !torch.vtensor<[64],f32>, !torch.vtensor<[64],f32>, !torch.vtensor<[384,64,1,1],f32>, !torch.vtensor<[384],f32>, !torch.vtensor<[384],f32>, !torch.vtensor<[384],f32>, !torch.vtensor<[384,1,3,3],f32>, !torch.vtensor<[384],f32>, !torch.vtensor<[384],f32>, !torch.vtensor<[384],f32>, !torch.vtensor<[64,384,1,1],f32>, !torch.vtensor<[64],f32>, !torch.vtensor<[64],f32>, !torch.vtensor<[64],f32>, !torch.vtensor<[384,64,1,1],f32>, !torch.vtensor<[384],f32>, !torch.vtensor<[384],f32>, !torch.vtensor<[384],f32>, !torch.vtensor<[384,1,3,3],f32>, !torch.vtensor<[384],f32>, !torch.vtensor<[384],f32>, !torch.vtensor<[384],f32>, !torch.vtensor<[64,384,1,1],f32>, !torch.vtensor<[64],f32>, !torch.vtensor<[64],f32>, !torch.vtensor<[64],f32>, !torch.vtensor<[384,64,1,1],f32>, !torch.vtensor<[384],f32>, !torch.vtensor<[384],f32>, !torch.vtensor<[384],f32>, !torch.vtensor<[384,1,3,3],f32>, !torch.vtensor<[384],f32>, !torch.vtensor<[384],f32>, !torch.vtensor<[384],f32>, !torch.vtensor<[96,384,1,1],f32>, !torch.vtensor<[96],f32>, !torch.vtensor<[96],f32>, !torch.vtensor<[96],f32>, !torch.vtensor<[576,96,1,1],f32>, !torch.vtensor<[576],f32>, !torch.vtensor<[576],f32>, !torch.vtensor<[576],f32>, !torch.vtensor<[576,1,3,3],f32>, !torch.vtensor<[576],f32>, !torch.vtensor<[576],f32>, !torch.vtensor<[576],f32>, !torch.vtensor<[96,576,1,1],f32>, !torch.vtensor<[96],f32>, !torch.vtensor<[96],f32>, !torch.vtensor<[96],f32>, !torch.vtensor<[576,96,1,1],f32>, !torch.vtensor<[576],f32>, !torch.vtensor<[576],f32>, !torch.vtensor<[576],f32>, !torch.vtensor<[576,1,3,3],f32>, !torch.vtensor<[576],f32>, !torch.vtensor<[576],f32>, !torch.vtensor<[576],f32>, !torch.vtensor<[96,576,1,1],f32>, !torch.vtensor<[96],f32>, !torch.vtensor<[96],f32>, !torch.vtensor<[96],f32>, !torch.vtensor<[576,96,1,1],f32>, !torch.vtensor<[576],f32>, !torch.vtensor<[576],f32>, !torch.vtensor<[576],f32>, !torch.vtensor<[576,1,3,3],f32>, !torch.vtensor<[576],f32>, !torch.vtensor<[576],f32>, !torch.vtensor<[576],f32>, !torch.vtensor<[160,576,1,1],f32>, !torch.vtensor<[160],f32>, !torch.vtensor<[160],f32>, !torch.vtensor<[160],f32>, !torch.vtensor<[960,160,1,1],f32>, !torch.vtensor<[960],f32>, !torch.vtensor<[960],f32>, !torch.vtensor<[960],f32>, !torch.vtensor<[960,1,3,3],f32>, !torch.vtensor<[960],f32>, !torch.vtensor<[960],f32>, !torch.vtensor<[960],f32>, !torch.vtensor<[160,960,1,1],f32>, !torch.vtensor<[160],f32>, !torch.vtensor<[160],f32>, !torch.vtensor<[160],f32>, !torch.vtensor<[960,160,1,1],f32>, !torch.vtensor<[960],f32>, !torch.vtensor<[960],f32>, !torch.vtensor<[960],f32>, !torch.vtensor<[960,1,3,3],f32>, !torch.vtensor<[960],f32>, !torch.vtensor<[960],f32>, !torch.vtensor<[960],f32>, !torch.vtensor<[160,960,1,1],f32>, !torch.vtensor<[160],f32>, !torch.vtensor<[160],f32>, !torch.vtensor<[160],f32>, !torch.vtensor<[960,160,1,1],f32>, !torch.vtensor<[960],f32>, !torch.vtensor<[960],f32>, !torch.vtensor<[960],f32>, !torch.vtensor<[960,1,3,3],f32>, !torch.vtensor<[960],f32>, !torch.vtensor<[960],f32>, !torch.vtensor<[960],f32>, !torch.vtensor<[320,960,1,1],f32>, !torch.vtensor<[320],f32>, !torch.vtensor<[320],f32>, !torch.vtensor<[320],f32>, !torch.vtensor<[1280,320,1,1],f32>, !torch.vtensor<[1280],f32>, !torch.vtensor<[1280],f32>, !torch.vtensor<[1280],f32>, !torch.vtensor<[1,3,225,225],f32>, !torch.vtensor<[1,32,112,112],f32>, !torch.vtensor<[1,32,112,112],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[1,32,114,114],f32>, !torch.vtensor<[1,32,112,112],f32>, !torch.vtensor<[1,32,112,112],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[1,32,112,112],f32>, !torch.vtensor<[1,16,112,112],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[1,16,112,112],f32>, !torch.vtensor<[1,96,112,112],f32>, !torch.vtensor<[1,96,112,112],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[1,96,113,113],f32>, !torch.vtensor<[1,96,56,56],f32>, !torch.vtensor<[1,96,56,56],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[1,96,56,56],f32>, !torch.vtensor<[1,24,56,56],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[1,24,56,56],f32>, !torch.vtensor<[1,144,56,56],f32>, !torch.vtensor<[1,144,56,56],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[1,144,58,58],f32>, !torch.vtensor<[1,144,56,56],f32>, !torch.vtensor<[1,144,56,56],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[1,144,56,56],f32>, !torch.vtensor<[1,24,56,56],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[1,24,56,56],f32>, !torch.vtensor<[1,144,56,56],f32>, !torch.vtensor<[1,144,56,56],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[1,144,57,57],f32>, !torch.vtensor<[1,144,28,28],f32>, !torch.vtensor<[1,144,28,28],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[1,144,28,28],f32>, !torch.vtensor<[1,32,28,28],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[1,32,28,28],f32>, !torch.vtensor<[1,192,28,28],f32>, !torch.vtensor<[1,192,28,28],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[1,192,30,30],f32>, !torch.vtensor<[1,192,28,28],f32>, !torch.vtensor<[1,192,28,28],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[1,192,28,28],f32>, !torch.vtensor<[1,32,28,28],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[1,32,28,28],f32>, !torch.vtensor<[1,192,28,28],f32>, !torch.vtensor<[1,192,28,28],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[1,192,30,30],f32>, !torch.vtensor<[1,192,28,28],f32>, !torch.vtensor<[1,192,28,28],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[1,192,28,28],f32>, !torch.vtensor<[1,32,28,28],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[1,32,28,28],f32>, !torch.vtensor<[1,192,28,28],f32>, !torch.vtensor<[1,192,28,28],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[1,192,29,29],f32>, !torch.vtensor<[1,192,14,14],f32>, !torch.vtensor<[1,192,14,14],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[1,192,14,14],f32>, !torch.vtensor<[1,64,14,14],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[1,64,14,14],f32>, !torch.vtensor<[1,384,14,14],f32>, !torch.vtensor<[1,384,14,14],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[1,384,16,16],f32>, !torch.vtensor<[1,384,14,14],f32>, !torch.vtensor<[1,384,14,14],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[1,384,14,14],f32>, !torch.vtensor<[1,64,14,14],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[1,64,14,14],f32>, !torch.vtensor<[1,384,14,14],f32>, !torch.vtensor<[1,384,14,14],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[1,384,16,16],f32>, !torch.vtensor<[1,384,14,14],f32>, !torch.vtensor<[1,384,14,14],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[1,384,14,14],f32>, !torch.vtensor<[1,64,14,14],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[1,64,14,14],f32>, !torch.vtensor<[1,384,14,14],f32>, !torch.vtensor<[1,384,14,14],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[1,384,16,16],f32>, !torch.vtensor<[1,384,14,14],f32>, !torch.vtensor<[1,384,14,14],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[1,384,14,14],f32>, !torch.vtensor<[1,64,14,14],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[1,64,14,14],f32>, !torch.vtensor<[1,384,14,14],f32>, !torch.vtensor<[1,384,14,14],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[1,384,16,16],f32>, !torch.vtensor<[1,384,14,14],f32>, !torch.vtensor<[1,384,14,14],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[1,384,14,14],f32>, !torch.vtensor<[1,96,14,14],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[1,96,14,14],f32>, !torch.vtensor<[1,576,14,14],f32>, !torch.vtensor<[1,576,14,14],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[1,576,16,16],f32>, !torch.vtensor<[1,576,14,14],f32>, !torch.vtensor<[1,576,14,14],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[1,576,14,14],f32>, !torch.vtensor<[1,96,14,14],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[1,96,14,14],f32>, !torch.vtensor<[1,576,14,14],f32>, !torch.vtensor<[1,576,14,14],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[1,576,16,16],f32>, !torch.vtensor<[1,576,14,14],f32>, !torch.vtensor<[1,576,14,14],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[1,576,14,14],f32>, !torch.vtensor<[1,96,14,14],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[1,96,14,14],f32>, !torch.vtensor<[1,576,14,14],f32>, !torch.vtensor<[1,576,14,14],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[1,576,15,15],f32>, !torch.vtensor<[1,576,7,7],f32>, !torch.vtensor<[1,576,7,7],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[1,576,7,7],f32>, !torch.vtensor<[1,160,7,7],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[1,160,7,7],f32>, !torch.vtensor<[1,960,7,7],f32>, !torch.vtensor<[1,960,7,7],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[1,960,9,9],f32>, !torch.vtensor<[1,960,7,7],f32>, !torch.vtensor<[1,960,7,7],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[1,960,7,7],f32>, !torch.vtensor<[1,160,7,7],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[1,160,7,7],f32>, !torch.vtensor<[1,960,7,7],f32>, !torch.vtensor<[1,960,7,7],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[1,960,9,9],f32>, !torch.vtensor<[1,960,7,7],f32>, !torch.vtensor<[1,960,7,7],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[1,960,7,7],f32>, !torch.vtensor<[1,160,7,7],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[1,160,7,7],f32>, !torch.vtensor<[1,960,7,7],f32>, !torch.vtensor<[1,960,7,7],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[1,960,9,9],f32>, !torch.vtensor<[1,960,7,7],f32>, !torch.vtensor<[1,960,7,7],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[1,960,7,7],f32>, !torch.vtensor<[1,320,7,7],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[1,320,7,7],f32>, !torch.vtensor<[1,1280,7,7],f32>, !torch.vtensor<[1,1280,7,7],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[1,1280],f32>, !torch.vtensor<[1280,1001],f32>
  }
}

#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d1, 0, 0)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map4 = affine_map<() -> ()>
#map5 = affine_map<(d0, d1, d2, d3) -> ()>
#map6 = affine_map<(d0, d1, d2, d3) -> (d0, d1, 0, 0)>
#map7 = affine_map<(d0, d1, d2, d3) -> (0, d1, 0, 0)>
#map8 = affine_map<(d0, d1) -> (0, d1)>
#map9 = affine_map<(d0, d1) -> (d0, d1)>
#map10 = affine_map<(d0, d1) -> (d1)>
module {
  util.func public @main$async(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view, %arg2: !hal.buffer_view, %arg3: !hal.buffer_view, %arg4: !hal.buffer_view, %arg5: !hal.buffer_view, %arg6: !hal.buffer_view, %arg7: !hal.buffer_view, %arg8: !hal.buffer_view, %arg9: !hal.buffer_view, %arg10: !hal.buffer_view, %arg11: !hal.buffer_view, %arg12: !hal.buffer_view, %arg13: !hal.buffer_view, %arg14: !hal.buffer_view, %arg15: !hal.buffer_view, %arg16: !hal.buffer_view, %arg17: !hal.buffer_view, %arg18: !hal.buffer_view, %arg19: !hal.buffer_view, %arg20: !hal.buffer_view, %arg21: !hal.buffer_view, %arg22: !hal.buffer_view, %arg23: !hal.buffer_view, %arg24: !hal.buffer_view, %arg25: !hal.buffer_view, %arg26: !hal.buffer_view, %arg27: !hal.buffer_view, %arg28: !hal.buffer_view, %arg29: !hal.buffer_view, %arg30: !hal.buffer_view, %arg31: !hal.buffer_view, %arg32: !hal.buffer_view, %arg33: !hal.buffer_view, %arg34: !hal.buffer_view, %arg35: !hal.buffer_view, %arg36: !hal.buffer_view, %arg37: !hal.buffer_view, %arg38: !hal.buffer_view, %arg39: !hal.buffer_view, %arg40: !hal.buffer_view, %arg41: !hal.buffer_view, %arg42: !hal.buffer_view, %arg43: !hal.buffer_view, %arg44: !hal.buffer_view, %arg45: !hal.buffer_view, %arg46: !hal.buffer_view, %arg47: !hal.buffer_view, %arg48: !hal.buffer_view, %arg49: !hal.buffer_view, %arg50: !hal.buffer_view, %arg51: !hal.buffer_view, %arg52: !hal.buffer_view, %arg53: !hal.buffer_view, %arg54: !hal.buffer_view, %arg55: !hal.buffer_view, %arg56: !hal.buffer_view, %arg57: !hal.buffer_view, %arg58: !hal.buffer_view, %arg59: !hal.buffer_view, %arg60: !hal.buffer_view, %arg61: !hal.buffer_view, %arg62: !hal.buffer_view, %arg63: !hal.buffer_view, %arg64: !hal.buffer_view, %arg65: !hal.buffer_view, %arg66: !hal.buffer_view, %arg67: !hal.buffer_view, %arg68: !hal.buffer_view, %arg69: !hal.buffer_view, %arg70: !hal.buffer_view, %arg71: !hal.buffer_view, %arg72: !hal.buffer_view, %arg73: !hal.buffer_view, %arg74: !hal.buffer_view, %arg75: !hal.buffer_view, %arg76: !hal.buffer_view, %arg77: !hal.buffer_view, %arg78: !hal.buffer_view, %arg79: !hal.buffer_view, %arg80: !hal.buffer_view, %arg81: !hal.buffer_view, %arg82: !hal.buffer_view, %arg83: !hal.buffer_view, %arg84: !hal.buffer_view, %arg85: !hal.buffer_view, %arg86: !hal.buffer_view, %arg87: !hal.buffer_view, %arg88: !hal.buffer_view, %arg89: !hal.buffer_view, %arg90: !hal.buffer_view, %arg91: !hal.buffer_view, %arg92: !hal.buffer_view, %arg93: !hal.buffer_view, %arg94: !hal.buffer_view, %arg95: !hal.buffer_view, %arg96: !hal.buffer_view, %arg97: !hal.buffer_view, %arg98: !hal.buffer_view, %arg99: !hal.buffer_view, %arg100: !hal.buffer_view, %arg101: !hal.buffer_view, %arg102: !hal.buffer_view, %arg103: !hal.buffer_view, %arg104: !hal.buffer_view, %arg105: !hal.buffer_view, %arg106: !hal.buffer_view, %arg107: !hal.buffer_view, %arg108: !hal.buffer_view, %arg109: !hal.buffer_view, %arg110: !hal.buffer_view, %arg111: !hal.buffer_view, %arg112: !hal.buffer_view, %arg113: !hal.buffer_view, %arg114: !hal.buffer_view, %arg115: !hal.buffer_view, %arg116: !hal.buffer_view, %arg117: !hal.buffer_view, %arg118: !hal.buffer_view, %arg119: !hal.buffer_view, %arg120: !hal.buffer_view, %arg121: !hal.buffer_view, %arg122: !hal.buffer_view, %arg123: !hal.buffer_view, %arg124: !hal.buffer_view, %arg125: !hal.buffer_view, %arg126: !hal.buffer_view, %arg127: !hal.buffer_view, %arg128: !hal.buffer_view, %arg129: !hal.buffer_view, %arg130: !hal.buffer_view, %arg131: !hal.buffer_view, %arg132: !hal.buffer_view, %arg133: !hal.buffer_view, %arg134: !hal.buffer_view, %arg135: !hal.buffer_view, %arg136: !hal.buffer_view, %arg137: !hal.buffer_view, %arg138: !hal.buffer_view, %arg139: !hal.buffer_view, %arg140: !hal.buffer_view, %arg141: !hal.buffer_view, %arg142: !hal.buffer_view, %arg143: !hal.buffer_view, %arg144: !hal.buffer_view, %arg145: !hal.buffer_view, %arg146: !hal.buffer_view, %arg147: !hal.buffer_view, %arg148: !hal.buffer_view, %arg149: !hal.buffer_view, %arg150: !hal.buffer_view, %arg151: !hal.buffer_view, %arg152: !hal.buffer_view, %arg153: !hal.buffer_view, %arg154: !hal.buffer_view, %arg155: !hal.buffer_view, %arg156: !hal.buffer_view, %arg157: !hal.buffer_view, %arg158: !hal.buffer_view, %arg159: !hal.buffer_view, %arg160: !hal.buffer_view, %arg161: !hal.buffer_view, %arg162: !hal.buffer_view, %arg163: !hal.buffer_view, %arg164: !hal.buffer_view, %arg165: !hal.buffer_view, %arg166: !hal.buffer_view, %arg167: !hal.buffer_view, %arg168: !hal.buffer_view, %arg169: !hal.buffer_view, %arg170: !hal.buffer_view, %arg171: !hal.buffer_view, %arg172: !hal.buffer_view, %arg173: !hal.buffer_view, %arg174: !hal.buffer_view, %arg175: !hal.buffer_view, %arg176: !hal.buffer_view, %arg177: !hal.buffer_view, %arg178: !hal.buffer_view, %arg179: !hal.buffer_view, %arg180: !hal.buffer_view, %arg181: !hal.buffer_view, %arg182: !hal.buffer_view, %arg183: !hal.buffer_view, %arg184: !hal.buffer_view, %arg185: !hal.buffer_view, %arg186: !hal.buffer_view, %arg187: !hal.buffer_view, %arg188: !hal.buffer_view, %arg189: !hal.buffer_view, %arg190: !hal.buffer_view, %arg191: !hal.buffer_view, %arg192: !hal.buffer_view, %arg193: !hal.buffer_view, %arg194: !hal.buffer_view, %arg195: !hal.buffer_view, %arg196: !hal.buffer_view, %arg197: !hal.buffer_view, %arg198: !hal.buffer_view, %arg199: !hal.buffer_view, %arg200: !hal.buffer_view, %arg201: !hal.buffer_view, %arg202: !hal.buffer_view, %arg203: !hal.buffer_view, %arg204: !hal.buffer_view, %arg205: !hal.buffer_view, %arg206: !hal.buffer_view, %arg207: !hal.buffer_view, %arg208: !hal.buffer_view, %arg209: !hal.buffer_view, %arg210: !hal.buffer_view, %arg211: !hal.buffer_view, %arg212: !hal.buffer_view, %arg213: !hal.buffer_view, %arg214: !hal.buffer_view, %arg215: !hal.buffer_view, %arg216: !hal.buffer_view, %arg217: !hal.buffer_view, %arg218: !hal.buffer_view, %arg219: !hal.buffer_view, %arg220: !hal.buffer_view, %arg221: !hal.buffer_view, %arg222: !hal.buffer_view, %arg223: !hal.buffer_view, %arg224: !hal.buffer_view, %arg225: !hal.buffer_view, %arg226: !hal.buffer_view, %arg227: !hal.buffer_view, %arg228: !hal.buffer_view, %arg229: !hal.buffer_view, %arg230: !hal.buffer_view, %arg231: !hal.buffer_view, %arg232: !hal.buffer_view, %arg233: !hal.buffer_view, %arg234: !hal.buffer_view, %arg235: !hal.buffer_view, %arg236: !hal.buffer_view, %arg237: !hal.buffer_view, %arg238: !hal.buffer_view, %arg239: !hal.buffer_view, %arg240: !hal.buffer_view, %arg241: !hal.buffer_view, %arg242: !hal.buffer_view, %arg243: !hal.buffer_view, %arg244: !hal.buffer_view, %arg245: !hal.buffer_view, %arg246: !hal.buffer_view, %arg247: !hal.buffer_view, %arg248: !hal.buffer_view, %arg249: !hal.buffer_view, %arg250: !hal.buffer_view, %arg251: !hal.buffer_view, %arg252: !hal.buffer_view, %arg253: !hal.buffer_view, %arg254: !hal.buffer_view, %arg255: !hal.buffer_view, %arg256: !hal.buffer_view, %arg257: !hal.buffer_view, %arg258: !hal.buffer_view, %arg259: !hal.buffer_view, %arg260: !hal.buffer_view, %arg261: !hal.buffer_view, %arg262: !hal.buffer_view, %arg263: !hal.fence, %arg264: !hal.fence) -> (!hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view) attributes {inlining_policy = #util.inline.never, iree.abi.model = "coarse-fences", iree.abi.stub} {
    %cst = arith.constant dense<6.000000e+00> : tensor<f64>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<f64>
    %cst_1 = arith.constant 0.000000e+00 : f32
    %cst_2 = arith.constant 1.000000e+00 : f32
    %cst_3 = arith.constant 1.000000e-03 : f64
    %cst_4 = arith.constant 4.900000e+01 : f32
    %0 = hal.tensor.import wait(%arg263) => %arg0 : !hal.buffer_view -> tensor<1x3x224x224xf32>
    %1 = hal.tensor.import wait(%arg263) => %arg1 : !hal.buffer_view -> tensor<32x3x3x3xf32>
    %2 = hal.tensor.import wait(%arg263) => %arg2 : !hal.buffer_view -> tensor<32xf32>
    %3 = hal.tensor.import wait(%arg263) => %arg3 : !hal.buffer_view -> tensor<32xf32>
    %4 = hal.tensor.import wait(%arg263) => %arg4 : !hal.buffer_view -> tensor<32xf32>
    %5 = hal.tensor.import wait(%arg263) => %arg5 : !hal.buffer_view -> tensor<32xf32>
    %6 = hal.tensor.import wait(%arg263) => %arg6 : !hal.buffer_view -> tensor<32x1x3x3xf32>
    %7 = hal.tensor.import wait(%arg263) => %arg7 : !hal.buffer_view -> tensor<32xf32>
    %8 = hal.tensor.import wait(%arg263) => %arg8 : !hal.buffer_view -> tensor<32xf32>
    %9 = hal.tensor.import wait(%arg263) => %arg9 : !hal.buffer_view -> tensor<32xf32>
    %10 = hal.tensor.import wait(%arg263) => %arg10 : !hal.buffer_view -> tensor<32xf32>
    %11 = hal.tensor.import wait(%arg263) => %arg11 : !hal.buffer_view -> tensor<16x32x1x1xf32>
    %12 = hal.tensor.import wait(%arg263) => %arg12 : !hal.buffer_view -> tensor<16xf32>
    %13 = hal.tensor.import wait(%arg263) => %arg13 : !hal.buffer_view -> tensor<16xf32>
    %14 = hal.tensor.import wait(%arg263) => %arg14 : !hal.buffer_view -> tensor<16xf32>
    %15 = hal.tensor.import wait(%arg263) => %arg15 : !hal.buffer_view -> tensor<16xf32>
    %16 = hal.tensor.import wait(%arg263) => %arg16 : !hal.buffer_view -> tensor<96x16x1x1xf32>
    %17 = hal.tensor.import wait(%arg263) => %arg17 : !hal.buffer_view -> tensor<96xf32>
    %18 = hal.tensor.import wait(%arg263) => %arg18 : !hal.buffer_view -> tensor<96xf32>
    %19 = hal.tensor.import wait(%arg263) => %arg19 : !hal.buffer_view -> tensor<96xf32>
    %20 = hal.tensor.import wait(%arg263) => %arg20 : !hal.buffer_view -> tensor<96xf32>
    %21 = hal.tensor.import wait(%arg263) => %arg21 : !hal.buffer_view -> tensor<96x1x3x3xf32>
    %22 = hal.tensor.import wait(%arg263) => %arg22 : !hal.buffer_view -> tensor<96xf32>
    %23 = hal.tensor.import wait(%arg263) => %arg23 : !hal.buffer_view -> tensor<96xf32>
    %24 = hal.tensor.import wait(%arg263) => %arg24 : !hal.buffer_view -> tensor<96xf32>
    %25 = hal.tensor.import wait(%arg263) => %arg25 : !hal.buffer_view -> tensor<96xf32>
    %26 = hal.tensor.import wait(%arg263) => %arg26 : !hal.buffer_view -> tensor<24x96x1x1xf32>
    %27 = hal.tensor.import wait(%arg263) => %arg27 : !hal.buffer_view -> tensor<24xf32>
    %28 = hal.tensor.import wait(%arg263) => %arg28 : !hal.buffer_view -> tensor<24xf32>
    %29 = hal.tensor.import wait(%arg263) => %arg29 : !hal.buffer_view -> tensor<24xf32>
    %30 = hal.tensor.import wait(%arg263) => %arg30 : !hal.buffer_view -> tensor<24xf32>
    %31 = hal.tensor.import wait(%arg263) => %arg31 : !hal.buffer_view -> tensor<144x24x1x1xf32>
    %32 = hal.tensor.import wait(%arg263) => %arg32 : !hal.buffer_view -> tensor<144xf32>
    %33 = hal.tensor.import wait(%arg263) => %arg33 : !hal.buffer_view -> tensor<144xf32>
    %34 = hal.tensor.import wait(%arg263) => %arg34 : !hal.buffer_view -> tensor<144xf32>
    %35 = hal.tensor.import wait(%arg263) => %arg35 : !hal.buffer_view -> tensor<144xf32>
    %36 = hal.tensor.import wait(%arg263) => %arg36 : !hal.buffer_view -> tensor<144x1x3x3xf32>
    %37 = hal.tensor.import wait(%arg263) => %arg37 : !hal.buffer_view -> tensor<144xf32>
    %38 = hal.tensor.import wait(%arg263) => %arg38 : !hal.buffer_view -> tensor<144xf32>
    %39 = hal.tensor.import wait(%arg263) => %arg39 : !hal.buffer_view -> tensor<144xf32>
    %40 = hal.tensor.import wait(%arg263) => %arg40 : !hal.buffer_view -> tensor<144xf32>
    %41 = hal.tensor.import wait(%arg263) => %arg41 : !hal.buffer_view -> tensor<24x144x1x1xf32>
    %42 = hal.tensor.import wait(%arg263) => %arg42 : !hal.buffer_view -> tensor<24xf32>
    %43 = hal.tensor.import wait(%arg263) => %arg43 : !hal.buffer_view -> tensor<24xf32>
    %44 = hal.tensor.import wait(%arg263) => %arg44 : !hal.buffer_view -> tensor<24xf32>
    %45 = hal.tensor.import wait(%arg263) => %arg45 : !hal.buffer_view -> tensor<24xf32>
    %46 = hal.tensor.import wait(%arg263) => %arg46 : !hal.buffer_view -> tensor<144x24x1x1xf32>
    %47 = hal.tensor.import wait(%arg263) => %arg47 : !hal.buffer_view -> tensor<144xf32>
    %48 = hal.tensor.import wait(%arg263) => %arg48 : !hal.buffer_view -> tensor<144xf32>
    %49 = hal.tensor.import wait(%arg263) => %arg49 : !hal.buffer_view -> tensor<144xf32>
    %50 = hal.tensor.import wait(%arg263) => %arg50 : !hal.buffer_view -> tensor<144xf32>
    %51 = hal.tensor.import wait(%arg263) => %arg51 : !hal.buffer_view -> tensor<144x1x3x3xf32>
    %52 = hal.tensor.import wait(%arg263) => %arg52 : !hal.buffer_view -> tensor<144xf32>
    %53 = hal.tensor.import wait(%arg263) => %arg53 : !hal.buffer_view -> tensor<144xf32>
    %54 = hal.tensor.import wait(%arg263) => %arg54 : !hal.buffer_view -> tensor<144xf32>
    %55 = hal.tensor.import wait(%arg263) => %arg55 : !hal.buffer_view -> tensor<144xf32>
    %56 = hal.tensor.import wait(%arg263) => %arg56 : !hal.buffer_view -> tensor<32x144x1x1xf32>
    %57 = hal.tensor.import wait(%arg263) => %arg57 : !hal.buffer_view -> tensor<32xf32>
    %58 = hal.tensor.import wait(%arg263) => %arg58 : !hal.buffer_view -> tensor<32xf32>
    %59 = hal.tensor.import wait(%arg263) => %arg59 : !hal.buffer_view -> tensor<32xf32>
    %60 = hal.tensor.import wait(%arg263) => %arg60 : !hal.buffer_view -> tensor<32xf32>
    %61 = hal.tensor.import wait(%arg263) => %arg61 : !hal.buffer_view -> tensor<192x32x1x1xf32>
    %62 = hal.tensor.import wait(%arg263) => %arg62 : !hal.buffer_view -> tensor<192xf32>
    %63 = hal.tensor.import wait(%arg263) => %arg63 : !hal.buffer_view -> tensor<192xf32>
    %64 = hal.tensor.import wait(%arg263) => %arg64 : !hal.buffer_view -> tensor<192xf32>
    %65 = hal.tensor.import wait(%arg263) => %arg65 : !hal.buffer_view -> tensor<192xf32>
    %66 = hal.tensor.import wait(%arg263) => %arg66 : !hal.buffer_view -> tensor<192x1x3x3xf32>
    %67 = hal.tensor.import wait(%arg263) => %arg67 : !hal.buffer_view -> tensor<192xf32>
    %68 = hal.tensor.import wait(%arg263) => %arg68 : !hal.buffer_view -> tensor<192xf32>
    %69 = hal.tensor.import wait(%arg263) => %arg69 : !hal.buffer_view -> tensor<192xf32>
    %70 = hal.tensor.import wait(%arg263) => %arg70 : !hal.buffer_view -> tensor<192xf32>
    %71 = hal.tensor.import wait(%arg263) => %arg71 : !hal.buffer_view -> tensor<32x192x1x1xf32>
    %72 = hal.tensor.import wait(%arg263) => %arg72 : !hal.buffer_view -> tensor<32xf32>
    %73 = hal.tensor.import wait(%arg263) => %arg73 : !hal.buffer_view -> tensor<32xf32>
    %74 = hal.tensor.import wait(%arg263) => %arg74 : !hal.buffer_view -> tensor<32xf32>
    %75 = hal.tensor.import wait(%arg263) => %arg75 : !hal.buffer_view -> tensor<32xf32>
    %76 = hal.tensor.import wait(%arg263) => %arg76 : !hal.buffer_view -> tensor<192x32x1x1xf32>
    %77 = hal.tensor.import wait(%arg263) => %arg77 : !hal.buffer_view -> tensor<192xf32>
    %78 = hal.tensor.import wait(%arg263) => %arg78 : !hal.buffer_view -> tensor<192xf32>
    %79 = hal.tensor.import wait(%arg263) => %arg79 : !hal.buffer_view -> tensor<192xf32>
    %80 = hal.tensor.import wait(%arg263) => %arg80 : !hal.buffer_view -> tensor<192xf32>
    %81 = hal.tensor.import wait(%arg263) => %arg81 : !hal.buffer_view -> tensor<192x1x3x3xf32>
    %82 = hal.tensor.import wait(%arg263) => %arg82 : !hal.buffer_view -> tensor<192xf32>
    %83 = hal.tensor.import wait(%arg263) => %arg83 : !hal.buffer_view -> tensor<192xf32>
    %84 = hal.tensor.import wait(%arg263) => %arg84 : !hal.buffer_view -> tensor<192xf32>
    %85 = hal.tensor.import wait(%arg263) => %arg85 : !hal.buffer_view -> tensor<192xf32>
    %86 = hal.tensor.import wait(%arg263) => %arg86 : !hal.buffer_view -> tensor<32x192x1x1xf32>
    %87 = hal.tensor.import wait(%arg263) => %arg87 : !hal.buffer_view -> tensor<32xf32>
    %88 = hal.tensor.import wait(%arg263) => %arg88 : !hal.buffer_view -> tensor<32xf32>
    %89 = hal.tensor.import wait(%arg263) => %arg89 : !hal.buffer_view -> tensor<32xf32>
    %90 = hal.tensor.import wait(%arg263) => %arg90 : !hal.buffer_view -> tensor<32xf32>
    %91 = hal.tensor.import wait(%arg263) => %arg91 : !hal.buffer_view -> tensor<192x32x1x1xf32>
    %92 = hal.tensor.import wait(%arg263) => %arg92 : !hal.buffer_view -> tensor<192xf32>
    %93 = hal.tensor.import wait(%arg263) => %arg93 : !hal.buffer_view -> tensor<192xf32>
    %94 = hal.tensor.import wait(%arg263) => %arg94 : !hal.buffer_view -> tensor<192xf32>
    %95 = hal.tensor.import wait(%arg263) => %arg95 : !hal.buffer_view -> tensor<192xf32>
    %96 = hal.tensor.import wait(%arg263) => %arg96 : !hal.buffer_view -> tensor<192x1x3x3xf32>
    %97 = hal.tensor.import wait(%arg263) => %arg97 : !hal.buffer_view -> tensor<192xf32>
    %98 = hal.tensor.import wait(%arg263) => %arg98 : !hal.buffer_view -> tensor<192xf32>
    %99 = hal.tensor.import wait(%arg263) => %arg99 : !hal.buffer_view -> tensor<192xf32>
    %100 = hal.tensor.import wait(%arg263) => %arg100 : !hal.buffer_view -> tensor<192xf32>
    %101 = hal.tensor.import wait(%arg263) => %arg101 : !hal.buffer_view -> tensor<64x192x1x1xf32>
    %102 = hal.tensor.import wait(%arg263) => %arg102 : !hal.buffer_view -> tensor<64xf32>
    %103 = hal.tensor.import wait(%arg263) => %arg103 : !hal.buffer_view -> tensor<64xf32>
    %104 = hal.tensor.import wait(%arg263) => %arg104 : !hal.buffer_view -> tensor<64xf32>
    %105 = hal.tensor.import wait(%arg263) => %arg105 : !hal.buffer_view -> tensor<64xf32>
    %106 = hal.tensor.import wait(%arg263) => %arg106 : !hal.buffer_view -> tensor<384x64x1x1xf32>
    %107 = hal.tensor.import wait(%arg263) => %arg107 : !hal.buffer_view -> tensor<384xf32>
    %108 = hal.tensor.import wait(%arg263) => %arg108 : !hal.buffer_view -> tensor<384xf32>
    %109 = hal.tensor.import wait(%arg263) => %arg109 : !hal.buffer_view -> tensor<384xf32>
    %110 = hal.tensor.import wait(%arg263) => %arg110 : !hal.buffer_view -> tensor<384xf32>
    %111 = hal.tensor.import wait(%arg263) => %arg111 : !hal.buffer_view -> tensor<384x1x3x3xf32>
    %112 = hal.tensor.import wait(%arg263) => %arg112 : !hal.buffer_view -> tensor<384xf32>
    %113 = hal.tensor.import wait(%arg263) => %arg113 : !hal.buffer_view -> tensor<384xf32>
    %114 = hal.tensor.import wait(%arg263) => %arg114 : !hal.buffer_view -> tensor<384xf32>
    %115 = hal.tensor.import wait(%arg263) => %arg115 : !hal.buffer_view -> tensor<384xf32>
    %116 = hal.tensor.import wait(%arg263) => %arg116 : !hal.buffer_view -> tensor<64x384x1x1xf32>
    %117 = hal.tensor.import wait(%arg263) => %arg117 : !hal.buffer_view -> tensor<64xf32>
    %118 = hal.tensor.import wait(%arg263) => %arg118 : !hal.buffer_view -> tensor<64xf32>
    %119 = hal.tensor.import wait(%arg263) => %arg119 : !hal.buffer_view -> tensor<64xf32>
    %120 = hal.tensor.import wait(%arg263) => %arg120 : !hal.buffer_view -> tensor<64xf32>
    %121 = hal.tensor.import wait(%arg263) => %arg121 : !hal.buffer_view -> tensor<384x64x1x1xf32>
    %122 = hal.tensor.import wait(%arg263) => %arg122 : !hal.buffer_view -> tensor<384xf32>
    %123 = hal.tensor.import wait(%arg263) => %arg123 : !hal.buffer_view -> tensor<384xf32>
    %124 = hal.tensor.import wait(%arg263) => %arg124 : !hal.buffer_view -> tensor<384xf32>
    %125 = hal.tensor.import wait(%arg263) => %arg125 : !hal.buffer_view -> tensor<384xf32>
    %126 = hal.tensor.import wait(%arg263) => %arg126 : !hal.buffer_view -> tensor<384x1x3x3xf32>
    %127 = hal.tensor.import wait(%arg263) => %arg127 : !hal.buffer_view -> tensor<384xf32>
    %128 = hal.tensor.import wait(%arg263) => %arg128 : !hal.buffer_view -> tensor<384xf32>
    %129 = hal.tensor.import wait(%arg263) => %arg129 : !hal.buffer_view -> tensor<384xf32>
    %130 = hal.tensor.import wait(%arg263) => %arg130 : !hal.buffer_view -> tensor<384xf32>
    %131 = hal.tensor.import wait(%arg263) => %arg131 : !hal.buffer_view -> tensor<64x384x1x1xf32>
    %132 = hal.tensor.import wait(%arg263) => %arg132 : !hal.buffer_view -> tensor<64xf32>
    %133 = hal.tensor.import wait(%arg263) => %arg133 : !hal.buffer_view -> tensor<64xf32>
    %134 = hal.tensor.import wait(%arg263) => %arg134 : !hal.buffer_view -> tensor<64xf32>
    %135 = hal.tensor.import wait(%arg263) => %arg135 : !hal.buffer_view -> tensor<64xf32>
    %136 = hal.tensor.import wait(%arg263) => %arg136 : !hal.buffer_view -> tensor<384x64x1x1xf32>
    %137 = hal.tensor.import wait(%arg263) => %arg137 : !hal.buffer_view -> tensor<384xf32>
    %138 = hal.tensor.import wait(%arg263) => %arg138 : !hal.buffer_view -> tensor<384xf32>
    %139 = hal.tensor.import wait(%arg263) => %arg139 : !hal.buffer_view -> tensor<384xf32>
    %140 = hal.tensor.import wait(%arg263) => %arg140 : !hal.buffer_view -> tensor<384xf32>
    %141 = hal.tensor.import wait(%arg263) => %arg141 : !hal.buffer_view -> tensor<384x1x3x3xf32>
    %142 = hal.tensor.import wait(%arg263) => %arg142 : !hal.buffer_view -> tensor<384xf32>
    %143 = hal.tensor.import wait(%arg263) => %arg143 : !hal.buffer_view -> tensor<384xf32>
    %144 = hal.tensor.import wait(%arg263) => %arg144 : !hal.buffer_view -> tensor<384xf32>
    %145 = hal.tensor.import wait(%arg263) => %arg145 : !hal.buffer_view -> tensor<384xf32>
    %146 = hal.tensor.import wait(%arg263) => %arg146 : !hal.buffer_view -> tensor<64x384x1x1xf32>
    %147 = hal.tensor.import wait(%arg263) => %arg147 : !hal.buffer_view -> tensor<64xf32>
    %148 = hal.tensor.import wait(%arg263) => %arg148 : !hal.buffer_view -> tensor<64xf32>
    %149 = hal.tensor.import wait(%arg263) => %arg149 : !hal.buffer_view -> tensor<64xf32>
    %150 = hal.tensor.import wait(%arg263) => %arg150 : !hal.buffer_view -> tensor<64xf32>
    %151 = hal.tensor.import wait(%arg263) => %arg151 : !hal.buffer_view -> tensor<384x64x1x1xf32>
    %152 = hal.tensor.import wait(%arg263) => %arg152 : !hal.buffer_view -> tensor<384xf32>
    %153 = hal.tensor.import wait(%arg263) => %arg153 : !hal.buffer_view -> tensor<384xf32>
    %154 = hal.tensor.import wait(%arg263) => %arg154 : !hal.buffer_view -> tensor<384xf32>
    %155 = hal.tensor.import wait(%arg263) => %arg155 : !hal.buffer_view -> tensor<384xf32>
    %156 = hal.tensor.import wait(%arg263) => %arg156 : !hal.buffer_view -> tensor<384x1x3x3xf32>
    %157 = hal.tensor.import wait(%arg263) => %arg157 : !hal.buffer_view -> tensor<384xf32>
    %158 = hal.tensor.import wait(%arg263) => %arg158 : !hal.buffer_view -> tensor<384xf32>
    %159 = hal.tensor.import wait(%arg263) => %arg159 : !hal.buffer_view -> tensor<384xf32>
    %160 = hal.tensor.import wait(%arg263) => %arg160 : !hal.buffer_view -> tensor<384xf32>
    %161 = hal.tensor.import wait(%arg263) => %arg161 : !hal.buffer_view -> tensor<96x384x1x1xf32>
    %162 = hal.tensor.import wait(%arg263) => %arg162 : !hal.buffer_view -> tensor<96xf32>
    %163 = hal.tensor.import wait(%arg263) => %arg163 : !hal.buffer_view -> tensor<96xf32>
    %164 = hal.tensor.import wait(%arg263) => %arg164 : !hal.buffer_view -> tensor<96xf32>
    %165 = hal.tensor.import wait(%arg263) => %arg165 : !hal.buffer_view -> tensor<96xf32>
    %166 = hal.tensor.import wait(%arg263) => %arg166 : !hal.buffer_view -> tensor<576x96x1x1xf32>
    %167 = hal.tensor.import wait(%arg263) => %arg167 : !hal.buffer_view -> tensor<576xf32>
    %168 = hal.tensor.import wait(%arg263) => %arg168 : !hal.buffer_view -> tensor<576xf32>
    %169 = hal.tensor.import wait(%arg263) => %arg169 : !hal.buffer_view -> tensor<576xf32>
    %170 = hal.tensor.import wait(%arg263) => %arg170 : !hal.buffer_view -> tensor<576xf32>
    %171 = hal.tensor.import wait(%arg263) => %arg171 : !hal.buffer_view -> tensor<576x1x3x3xf32>
    %172 = hal.tensor.import wait(%arg263) => %arg172 : !hal.buffer_view -> tensor<576xf32>
    %173 = hal.tensor.import wait(%arg263) => %arg173 : !hal.buffer_view -> tensor<576xf32>
    %174 = hal.tensor.import wait(%arg263) => %arg174 : !hal.buffer_view -> tensor<576xf32>
    %175 = hal.tensor.import wait(%arg263) => %arg175 : !hal.buffer_view -> tensor<576xf32>
    %176 = hal.tensor.import wait(%arg263) => %arg176 : !hal.buffer_view -> tensor<96x576x1x1xf32>
    %177 = hal.tensor.import wait(%arg263) => %arg177 : !hal.buffer_view -> tensor<96xf32>
    %178 = hal.tensor.import wait(%arg263) => %arg178 : !hal.buffer_view -> tensor<96xf32>
    %179 = hal.tensor.import wait(%arg263) => %arg179 : !hal.buffer_view -> tensor<96xf32>
    %180 = hal.tensor.import wait(%arg263) => %arg180 : !hal.buffer_view -> tensor<96xf32>
    %181 = hal.tensor.import wait(%arg263) => %arg181 : !hal.buffer_view -> tensor<576x96x1x1xf32>
    %182 = hal.tensor.import wait(%arg263) => %arg182 : !hal.buffer_view -> tensor<576xf32>
    %183 = hal.tensor.import wait(%arg263) => %arg183 : !hal.buffer_view -> tensor<576xf32>
    %184 = hal.tensor.import wait(%arg263) => %arg184 : !hal.buffer_view -> tensor<576xf32>
    %185 = hal.tensor.import wait(%arg263) => %arg185 : !hal.buffer_view -> tensor<576xf32>
    %186 = hal.tensor.import wait(%arg263) => %arg186 : !hal.buffer_view -> tensor<576x1x3x3xf32>
    %187 = hal.tensor.import wait(%arg263) => %arg187 : !hal.buffer_view -> tensor<576xf32>
    %188 = hal.tensor.import wait(%arg263) => %arg188 : !hal.buffer_view -> tensor<576xf32>
    %189 = hal.tensor.import wait(%arg263) => %arg189 : !hal.buffer_view -> tensor<576xf32>
    %190 = hal.tensor.import wait(%arg263) => %arg190 : !hal.buffer_view -> tensor<576xf32>
    %191 = hal.tensor.import wait(%arg263) => %arg191 : !hal.buffer_view -> tensor<96x576x1x1xf32>
    %192 = hal.tensor.import wait(%arg263) => %arg192 : !hal.buffer_view -> tensor<96xf32>
    %193 = hal.tensor.import wait(%arg263) => %arg193 : !hal.buffer_view -> tensor<96xf32>
    %194 = hal.tensor.import wait(%arg263) => %arg194 : !hal.buffer_view -> tensor<96xf32>
    %195 = hal.tensor.import wait(%arg263) => %arg195 : !hal.buffer_view -> tensor<96xf32>
    %196 = hal.tensor.import wait(%arg263) => %arg196 : !hal.buffer_view -> tensor<576x96x1x1xf32>
    %197 = hal.tensor.import wait(%arg263) => %arg197 : !hal.buffer_view -> tensor<576xf32>
    %198 = hal.tensor.import wait(%arg263) => %arg198 : !hal.buffer_view -> tensor<576xf32>
    %199 = hal.tensor.import wait(%arg263) => %arg199 : !hal.buffer_view -> tensor<576xf32>
    %200 = hal.tensor.import wait(%arg263) => %arg200 : !hal.buffer_view -> tensor<576xf32>
    %201 = hal.tensor.import wait(%arg263) => %arg201 : !hal.buffer_view -> tensor<576x1x3x3xf32>
    %202 = hal.tensor.import wait(%arg263) => %arg202 : !hal.buffer_view -> tensor<576xf32>
    %203 = hal.tensor.import wait(%arg263) => %arg203 : !hal.buffer_view -> tensor<576xf32>
    %204 = hal.tensor.import wait(%arg263) => %arg204 : !hal.buffer_view -> tensor<576xf32>
    %205 = hal.tensor.import wait(%arg263) => %arg205 : !hal.buffer_view -> tensor<576xf32>
    %206 = hal.tensor.import wait(%arg263) => %arg206 : !hal.buffer_view -> tensor<160x576x1x1xf32>
    %207 = hal.tensor.import wait(%arg263) => %arg207 : !hal.buffer_view -> tensor<160xf32>
    %208 = hal.tensor.import wait(%arg263) => %arg208 : !hal.buffer_view -> tensor<160xf32>
    %209 = hal.tensor.import wait(%arg263) => %arg209 : !hal.buffer_view -> tensor<160xf32>
    %210 = hal.tensor.import wait(%arg263) => %arg210 : !hal.buffer_view -> tensor<160xf32>
    %211 = hal.tensor.import wait(%arg263) => %arg211 : !hal.buffer_view -> tensor<960x160x1x1xf32>
    %212 = hal.tensor.import wait(%arg263) => %arg212 : !hal.buffer_view -> tensor<960xf32>
    %213 = hal.tensor.import wait(%arg263) => %arg213 : !hal.buffer_view -> tensor<960xf32>
    %214 = hal.tensor.import wait(%arg263) => %arg214 : !hal.buffer_view -> tensor<960xf32>
    %215 = hal.tensor.import wait(%arg263) => %arg215 : !hal.buffer_view -> tensor<960xf32>
    %216 = hal.tensor.import wait(%arg263) => %arg216 : !hal.buffer_view -> tensor<960x1x3x3xf32>
    %217 = hal.tensor.import wait(%arg263) => %arg217 : !hal.buffer_view -> tensor<960xf32>
    %218 = hal.tensor.import wait(%arg263) => %arg218 : !hal.buffer_view -> tensor<960xf32>
    %219 = hal.tensor.import wait(%arg263) => %arg219 : !hal.buffer_view -> tensor<960xf32>
    %220 = hal.tensor.import wait(%arg263) => %arg220 : !hal.buffer_view -> tensor<960xf32>
    %221 = hal.tensor.import wait(%arg263) => %arg221 : !hal.buffer_view -> tensor<160x960x1x1xf32>
    %222 = hal.tensor.import wait(%arg263) => %arg222 : !hal.buffer_view -> tensor<160xf32>
    %223 = hal.tensor.import wait(%arg263) => %arg223 : !hal.buffer_view -> tensor<160xf32>
    %224 = hal.tensor.import wait(%arg263) => %arg224 : !hal.buffer_view -> tensor<160xf32>
    %225 = hal.tensor.import wait(%arg263) => %arg225 : !hal.buffer_view -> tensor<160xf32>
    %226 = hal.tensor.import wait(%arg263) => %arg226 : !hal.buffer_view -> tensor<960x160x1x1xf32>
    %227 = hal.tensor.import wait(%arg263) => %arg227 : !hal.buffer_view -> tensor<960xf32>
    %228 = hal.tensor.import wait(%arg263) => %arg228 : !hal.buffer_view -> tensor<960xf32>
    %229 = hal.tensor.import wait(%arg263) => %arg229 : !hal.buffer_view -> tensor<960xf32>
    %230 = hal.tensor.import wait(%arg263) => %arg230 : !hal.buffer_view -> tensor<960xf32>
    %231 = hal.tensor.import wait(%arg263) => %arg231 : !hal.buffer_view -> tensor<960x1x3x3xf32>
    %232 = hal.tensor.import wait(%arg263) => %arg232 : !hal.buffer_view -> tensor<960xf32>
    %233 = hal.tensor.import wait(%arg263) => %arg233 : !hal.buffer_view -> tensor<960xf32>
    %234 = hal.tensor.import wait(%arg263) => %arg234 : !hal.buffer_view -> tensor<960xf32>
    %235 = hal.tensor.import wait(%arg263) => %arg235 : !hal.buffer_view -> tensor<960xf32>
    %236 = hal.tensor.import wait(%arg263) => %arg236 : !hal.buffer_view -> tensor<160x960x1x1xf32>
    %237 = hal.tensor.import wait(%arg263) => %arg237 : !hal.buffer_view -> tensor<160xf32>
    %238 = hal.tensor.import wait(%arg263) => %arg238 : !hal.buffer_view -> tensor<160xf32>
    %239 = hal.tensor.import wait(%arg263) => %arg239 : !hal.buffer_view -> tensor<160xf32>
    %240 = hal.tensor.import wait(%arg263) => %arg240 : !hal.buffer_view -> tensor<160xf32>
    %241 = hal.tensor.import wait(%arg263) => %arg241 : !hal.buffer_view -> tensor<960x160x1x1xf32>
    %242 = hal.tensor.import wait(%arg263) => %arg242 : !hal.buffer_view -> tensor<960xf32>
    %243 = hal.tensor.import wait(%arg263) => %arg243 : !hal.buffer_view -> tensor<960xf32>
    %244 = hal.tensor.import wait(%arg263) => %arg244 : !hal.buffer_view -> tensor<960xf32>
    %245 = hal.tensor.import wait(%arg263) => %arg245 : !hal.buffer_view -> tensor<960xf32>
    %246 = hal.tensor.import wait(%arg263) => %arg246 : !hal.buffer_view -> tensor<960x1x3x3xf32>
    %247 = hal.tensor.import wait(%arg263) => %arg247 : !hal.buffer_view -> tensor<960xf32>
    %248 = hal.tensor.import wait(%arg263) => %arg248 : !hal.buffer_view -> tensor<960xf32>
    %249 = hal.tensor.import wait(%arg263) => %arg249 : !hal.buffer_view -> tensor<960xf32>
    %250 = hal.tensor.import wait(%arg263) => %arg250 : !hal.buffer_view -> tensor<960xf32>
    %251 = hal.tensor.import wait(%arg263) => %arg251 : !hal.buffer_view -> tensor<320x960x1x1xf32>
    %252 = hal.tensor.import wait(%arg263) => %arg252 : !hal.buffer_view -> tensor<320xf32>
    %253 = hal.tensor.import wait(%arg263) => %arg253 : !hal.buffer_view -> tensor<320xf32>
    %254 = hal.tensor.import wait(%arg263) => %arg254 : !hal.buffer_view -> tensor<320xf32>
    %255 = hal.tensor.import wait(%arg263) => %arg255 : !hal.buffer_view -> tensor<320xf32>
    %256 = hal.tensor.import wait(%arg263) => %arg256 : !hal.buffer_view -> tensor<1280x320x1x1xf32>
    %257 = hal.tensor.import wait(%arg263) => %arg257 : !hal.buffer_view -> tensor<1280xf32>
    %258 = hal.tensor.import wait(%arg263) => %arg258 : !hal.buffer_view -> tensor<1280xf32>
    %259 = hal.tensor.import wait(%arg263) => %arg259 : !hal.buffer_view -> tensor<1280xf32>
    %260 = hal.tensor.import wait(%arg263) => %arg260 : !hal.buffer_view -> tensor<1280xf32>
    %261 = hal.tensor.import wait(%arg263) => %arg261 : !hal.buffer_view -> tensor<1001x1280xf32>
    %262 = hal.tensor.import wait(%arg263) => %arg262 : !hal.buffer_view -> tensor<1001xf32>
    %padded = tensor.pad %0 low[0, 0, 0, 0] high[0, 0, 1, 1] {
    ^bb0(%arg265: index, %arg266: index, %arg267: index, %arg268: index):
      tensor.yield %cst_1 : f32
    } : tensor<1x3x224x224xf32> to tensor<1x3x225x225xf32>
    %263 = tensor.empty() : tensor<1x32x112x112xf32>
    %264 = linalg.fill ins(%cst_1 : f32) outs(%263 : tensor<1x32x112x112xf32>) -> tensor<1x32x112x112xf32>
    %265 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%padded, %1 : tensor<1x3x225x225xf32>, tensor<32x3x3x3xf32>) outs(%264 : tensor<1x32x112x112xf32>) -> tensor<1x32x112x112xf32>
    %266 = tensor.empty() : tensor<32xf32>
    %267 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%3 : tensor<32xf32>) outs(%266 : tensor<32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = arith.truncf %cst_3 : f64 to f32
      %1281 = arith.addf %in, %1280 : f32
      linalg.yield %1281 : f32
    } -> tensor<32xf32>
    %268 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%267 : tensor<32xf32>) outs(%266 : tensor<32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = math.sqrt %in : f32
      linalg.yield %1280 : f32
    } -> tensor<32xf32>
    %269 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%268 : tensor<32xf32>) outs(%266 : tensor<32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = arith.cmpf one, %in, %cst_1 : f32
      cf.assert %1280, "unimplemented: tensor with zero element"
      %1281 = arith.divf %cst_2, %in : f32
      linalg.yield %1281 : f32
    } -> tensor<32xf32>
    %270 = tensor.empty() : tensor<0xf32>
    %271 = linalg.fill ins(%cst_1 : f32) outs(%270 : tensor<0xf32>) -> tensor<0xf32>
    %expanded = tensor.expand_shape %2 [[0, 1, 2]] output_shape [32, 1, 1] : tensor<32xf32> into tensor<32x1x1xf32>
    %expanded_5 = tensor.expand_shape %269 [[0, 1, 2]] output_shape [32, 1, 1] : tensor<32xf32> into tensor<32x1x1xf32>
    %272 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%265, %expanded : tensor<1x32x112x112xf32>, tensor<32x1x1xf32>) outs(%263 : tensor<1x32x112x112xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.subf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x32x112x112xf32>
    %273 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%272, %expanded_5 : tensor<1x32x112x112xf32>, tensor<32x1x1xf32>) outs(%263 : tensor<1x32x112x112xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.mulf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x32x112x112xf32>
    %expanded_6 = tensor.expand_shape %4 [[0, 1, 2]] output_shape [32, 1, 1] : tensor<32xf32> into tensor<32x1x1xf32>
    %274 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%273, %expanded_6 : tensor<1x32x112x112xf32>, tensor<32x1x1xf32>) outs(%263 : tensor<1x32x112x112xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.mulf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x32x112x112xf32>
    %expanded_7 = tensor.expand_shape %5 [[0, 1, 2]] output_shape [32, 1, 1] : tensor<32xf32> into tensor<32x1x1xf32>
    %275 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%274, %expanded_7 : tensor<1x32x112x112xf32>, tensor<32x1x1xf32>) outs(%263 : tensor<1x32x112x112xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.addf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x32x112x112xf32>
    %276 = tensor.empty() : tensor<f32>
    %277 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = []} ins(%cst_0 : tensor<f64>) outs(%276 : tensor<f32>) {
    ^bb0(%in: f64, %out: f32):
      %1280 = arith.truncf %in : f64 to f32
      linalg.yield %1280 : f32
    } -> tensor<f32>
    %278 = linalg.generic {indexing_maps = [#map1, #map5, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%275, %277 : tensor<1x32x112x112xf32>, tensor<f32>) outs(%263 : tensor<1x32x112x112xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.cmpf ogt, %in, %in_246 : f32
      %1281 = arith.select %1280, %in, %in_246 : f32
      linalg.yield %1281 : f32
    } -> tensor<1x32x112x112xf32>
    %279 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = []} ins(%cst : tensor<f64>) outs(%276 : tensor<f32>) {
    ^bb0(%in: f64, %out: f32):
      %1280 = arith.truncf %in : f64 to f32
      linalg.yield %1280 : f32
    } -> tensor<f32>
    %280 = linalg.generic {indexing_maps = [#map5, #map1, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%279, %278 : tensor<f32>, tensor<1x32x112x112xf32>) outs(%263 : tensor<1x32x112x112xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.cmpf olt, %in, %in_246 : f32
      %1281 = arith.select %1280, %in, %in_246 : f32
      linalg.yield %1281 : f32
    } -> tensor<1x32x112x112xf32>
    %padded_8 = tensor.pad %280 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg265: index, %arg266: index, %arg267: index, %arg268: index):
      tensor.yield %cst_1 : f32
    } : tensor<1x32x112x112xf32> to tensor<1x32x114x114xf32>
    %collapsed = tensor.collapse_shape %6 [[0, 1], [2], [3]] : tensor<32x1x3x3xf32> into tensor<32x3x3xf32>
    %281 = linalg.depthwise_conv_2d_nchw_chw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_8, %collapsed : tensor<1x32x114x114xf32>, tensor<32x3x3xf32>) outs(%264 : tensor<1x32x112x112xf32>) -> tensor<1x32x112x112xf32>
    %282 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%8 : tensor<32xf32>) outs(%266 : tensor<32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = arith.truncf %cst_3 : f64 to f32
      %1281 = arith.addf %in, %1280 : f32
      linalg.yield %1281 : f32
    } -> tensor<32xf32>
    %283 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%282 : tensor<32xf32>) outs(%266 : tensor<32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = math.sqrt %in : f32
      linalg.yield %1280 : f32
    } -> tensor<32xf32>
    %284 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%283 : tensor<32xf32>) outs(%266 : tensor<32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = arith.cmpf one, %in, %cst_1 : f32
      cf.assert %1280, "unimplemented: tensor with zero element"
      %1281 = arith.divf %cst_2, %in : f32
      linalg.yield %1281 : f32
    } -> tensor<32xf32>
    %expanded_9 = tensor.expand_shape %7 [[0, 1, 2]] output_shape [32, 1, 1] : tensor<32xf32> into tensor<32x1x1xf32>
    %expanded_10 = tensor.expand_shape %284 [[0, 1, 2]] output_shape [32, 1, 1] : tensor<32xf32> into tensor<32x1x1xf32>
    %285 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%281, %expanded_9 : tensor<1x32x112x112xf32>, tensor<32x1x1xf32>) outs(%263 : tensor<1x32x112x112xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.subf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x32x112x112xf32>
    %286 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%285, %expanded_10 : tensor<1x32x112x112xf32>, tensor<32x1x1xf32>) outs(%263 : tensor<1x32x112x112xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.mulf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x32x112x112xf32>
    %expanded_11 = tensor.expand_shape %9 [[0, 1, 2]] output_shape [32, 1, 1] : tensor<32xf32> into tensor<32x1x1xf32>
    %287 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%286, %expanded_11 : tensor<1x32x112x112xf32>, tensor<32x1x1xf32>) outs(%263 : tensor<1x32x112x112xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.mulf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x32x112x112xf32>
    %expanded_12 = tensor.expand_shape %10 [[0, 1, 2]] output_shape [32, 1, 1] : tensor<32xf32> into tensor<32x1x1xf32>
    %288 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%287, %expanded_12 : tensor<1x32x112x112xf32>, tensor<32x1x1xf32>) outs(%263 : tensor<1x32x112x112xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.addf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x32x112x112xf32>
    %289 = linalg.generic {indexing_maps = [#map1, #map5, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%288, %277 : tensor<1x32x112x112xf32>, tensor<f32>) outs(%263 : tensor<1x32x112x112xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.cmpf ogt, %in, %in_246 : f32
      %1281 = arith.select %1280, %in, %in_246 : f32
      linalg.yield %1281 : f32
    } -> tensor<1x32x112x112xf32>
    %290 = linalg.generic {indexing_maps = [#map5, #map1, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%279, %289 : tensor<f32>, tensor<1x32x112x112xf32>) outs(%263 : tensor<1x32x112x112xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.cmpf olt, %in, %in_246 : f32
      %1281 = arith.select %1280, %in, %in_246 : f32
      linalg.yield %1281 : f32
    } -> tensor<1x32x112x112xf32>
    %291 = tensor.empty() : tensor<1x16x112x112xf32>
    %292 = linalg.fill ins(%cst_1 : f32) outs(%291 : tensor<1x16x112x112xf32>) -> tensor<1x16x112x112xf32>
    %293 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%290, %11 : tensor<1x32x112x112xf32>, tensor<16x32x1x1xf32>) outs(%292 : tensor<1x16x112x112xf32>) -> tensor<1x16x112x112xf32>
    %294 = tensor.empty() : tensor<16xf32>
    %295 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%13 : tensor<16xf32>) outs(%294 : tensor<16xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = arith.truncf %cst_3 : f64 to f32
      %1281 = arith.addf %in, %1280 : f32
      linalg.yield %1281 : f32
    } -> tensor<16xf32>
    %296 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%295 : tensor<16xf32>) outs(%294 : tensor<16xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = math.sqrt %in : f32
      linalg.yield %1280 : f32
    } -> tensor<16xf32>
    %297 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%296 : tensor<16xf32>) outs(%294 : tensor<16xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = arith.cmpf one, %in, %cst_1 : f32
      cf.assert %1280, "unimplemented: tensor with zero element"
      %1281 = arith.divf %cst_2, %in : f32
      linalg.yield %1281 : f32
    } -> tensor<16xf32>
    %expanded_13 = tensor.expand_shape %12 [[0, 1, 2]] output_shape [16, 1, 1] : tensor<16xf32> into tensor<16x1x1xf32>
    %expanded_14 = tensor.expand_shape %297 [[0, 1, 2]] output_shape [16, 1, 1] : tensor<16xf32> into tensor<16x1x1xf32>
    %298 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%293, %expanded_13 : tensor<1x16x112x112xf32>, tensor<16x1x1xf32>) outs(%291 : tensor<1x16x112x112xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.subf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x16x112x112xf32>
    %299 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%298, %expanded_14 : tensor<1x16x112x112xf32>, tensor<16x1x1xf32>) outs(%291 : tensor<1x16x112x112xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.mulf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x16x112x112xf32>
    %expanded_15 = tensor.expand_shape %14 [[0, 1, 2]] output_shape [16, 1, 1] : tensor<16xf32> into tensor<16x1x1xf32>
    %300 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%299, %expanded_15 : tensor<1x16x112x112xf32>, tensor<16x1x1xf32>) outs(%291 : tensor<1x16x112x112xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.mulf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x16x112x112xf32>
    %expanded_16 = tensor.expand_shape %15 [[0, 1, 2]] output_shape [16, 1, 1] : tensor<16xf32> into tensor<16x1x1xf32>
    %301 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%300, %expanded_16 : tensor<1x16x112x112xf32>, tensor<16x1x1xf32>) outs(%291 : tensor<1x16x112x112xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.addf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x16x112x112xf32>
    %302 = tensor.empty() : tensor<1x96x112x112xf32>
    %303 = linalg.fill ins(%cst_1 : f32) outs(%302 : tensor<1x96x112x112xf32>) -> tensor<1x96x112x112xf32>
    %304 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%301, %16 : tensor<1x16x112x112xf32>, tensor<96x16x1x1xf32>) outs(%303 : tensor<1x96x112x112xf32>) -> tensor<1x96x112x112xf32>
    %305 = tensor.empty() : tensor<96xf32>
    %306 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%18 : tensor<96xf32>) outs(%305 : tensor<96xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = arith.truncf %cst_3 : f64 to f32
      %1281 = arith.addf %in, %1280 : f32
      linalg.yield %1281 : f32
    } -> tensor<96xf32>
    %307 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%306 : tensor<96xf32>) outs(%305 : tensor<96xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = math.sqrt %in : f32
      linalg.yield %1280 : f32
    } -> tensor<96xf32>
    %308 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%307 : tensor<96xf32>) outs(%305 : tensor<96xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = arith.cmpf one, %in, %cst_1 : f32
      cf.assert %1280, "unimplemented: tensor with zero element"
      %1281 = arith.divf %cst_2, %in : f32
      linalg.yield %1281 : f32
    } -> tensor<96xf32>
    %expanded_17 = tensor.expand_shape %17 [[0, 1, 2]] output_shape [96, 1, 1] : tensor<96xf32> into tensor<96x1x1xf32>
    %expanded_18 = tensor.expand_shape %308 [[0, 1, 2]] output_shape [96, 1, 1] : tensor<96xf32> into tensor<96x1x1xf32>
    %309 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%304, %expanded_17 : tensor<1x96x112x112xf32>, tensor<96x1x1xf32>) outs(%302 : tensor<1x96x112x112xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.subf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x96x112x112xf32>
    %310 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%309, %expanded_18 : tensor<1x96x112x112xf32>, tensor<96x1x1xf32>) outs(%302 : tensor<1x96x112x112xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.mulf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x96x112x112xf32>
    %expanded_19 = tensor.expand_shape %19 [[0, 1, 2]] output_shape [96, 1, 1] : tensor<96xf32> into tensor<96x1x1xf32>
    %311 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%310, %expanded_19 : tensor<1x96x112x112xf32>, tensor<96x1x1xf32>) outs(%302 : tensor<1x96x112x112xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.mulf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x96x112x112xf32>
    %expanded_20 = tensor.expand_shape %20 [[0, 1, 2]] output_shape [96, 1, 1] : tensor<96xf32> into tensor<96x1x1xf32>
    %312 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%311, %expanded_20 : tensor<1x96x112x112xf32>, tensor<96x1x1xf32>) outs(%302 : tensor<1x96x112x112xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.addf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x96x112x112xf32>
    %313 = linalg.generic {indexing_maps = [#map1, #map5, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%312, %277 : tensor<1x96x112x112xf32>, tensor<f32>) outs(%302 : tensor<1x96x112x112xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.cmpf ogt, %in, %in_246 : f32
      %1281 = arith.select %1280, %in, %in_246 : f32
      linalg.yield %1281 : f32
    } -> tensor<1x96x112x112xf32>
    %314 = linalg.generic {indexing_maps = [#map5, #map1, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%279, %313 : tensor<f32>, tensor<1x96x112x112xf32>) outs(%302 : tensor<1x96x112x112xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.cmpf olt, %in, %in_246 : f32
      %1281 = arith.select %1280, %in, %in_246 : f32
      linalg.yield %1281 : f32
    } -> tensor<1x96x112x112xf32>
    %padded_21 = tensor.pad %314 low[0, 0, 0, 0] high[0, 0, 1, 1] {
    ^bb0(%arg265: index, %arg266: index, %arg267: index, %arg268: index):
      tensor.yield %cst_1 : f32
    } : tensor<1x96x112x112xf32> to tensor<1x96x113x113xf32>
    %315 = tensor.empty() : tensor<1x96x56x56xf32>
    %316 = linalg.fill ins(%cst_1 : f32) outs(%315 : tensor<1x96x56x56xf32>) -> tensor<1x96x56x56xf32>
    %collapsed_22 = tensor.collapse_shape %21 [[0, 1], [2], [3]] : tensor<96x1x3x3xf32> into tensor<96x3x3xf32>
    %317 = linalg.depthwise_conv_2d_nchw_chw {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%padded_21, %collapsed_22 : tensor<1x96x113x113xf32>, tensor<96x3x3xf32>) outs(%316 : tensor<1x96x56x56xf32>) -> tensor<1x96x56x56xf32>
    %318 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%23 : tensor<96xf32>) outs(%305 : tensor<96xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = arith.truncf %cst_3 : f64 to f32
      %1281 = arith.addf %in, %1280 : f32
      linalg.yield %1281 : f32
    } -> tensor<96xf32>
    %319 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%318 : tensor<96xf32>) outs(%305 : tensor<96xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = math.sqrt %in : f32
      linalg.yield %1280 : f32
    } -> tensor<96xf32>
    %320 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%319 : tensor<96xf32>) outs(%305 : tensor<96xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = arith.cmpf one, %in, %cst_1 : f32
      cf.assert %1280, "unimplemented: tensor with zero element"
      %1281 = arith.divf %cst_2, %in : f32
      linalg.yield %1281 : f32
    } -> tensor<96xf32>
    %expanded_23 = tensor.expand_shape %22 [[0, 1, 2]] output_shape [96, 1, 1] : tensor<96xf32> into tensor<96x1x1xf32>
    %expanded_24 = tensor.expand_shape %320 [[0, 1, 2]] output_shape [96, 1, 1] : tensor<96xf32> into tensor<96x1x1xf32>
    %321 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%317, %expanded_23 : tensor<1x96x56x56xf32>, tensor<96x1x1xf32>) outs(%315 : tensor<1x96x56x56xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.subf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x96x56x56xf32>
    %322 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%321, %expanded_24 : tensor<1x96x56x56xf32>, tensor<96x1x1xf32>) outs(%315 : tensor<1x96x56x56xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.mulf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x96x56x56xf32>
    %expanded_25 = tensor.expand_shape %24 [[0, 1, 2]] output_shape [96, 1, 1] : tensor<96xf32> into tensor<96x1x1xf32>
    %323 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%322, %expanded_25 : tensor<1x96x56x56xf32>, tensor<96x1x1xf32>) outs(%315 : tensor<1x96x56x56xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.mulf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x96x56x56xf32>
    %expanded_26 = tensor.expand_shape %25 [[0, 1, 2]] output_shape [96, 1, 1] : tensor<96xf32> into tensor<96x1x1xf32>
    %324 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%323, %expanded_26 : tensor<1x96x56x56xf32>, tensor<96x1x1xf32>) outs(%315 : tensor<1x96x56x56xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.addf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x96x56x56xf32>
    %325 = linalg.generic {indexing_maps = [#map1, #map5, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%324, %277 : tensor<1x96x56x56xf32>, tensor<f32>) outs(%315 : tensor<1x96x56x56xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.cmpf ogt, %in, %in_246 : f32
      %1281 = arith.select %1280, %in, %in_246 : f32
      linalg.yield %1281 : f32
    } -> tensor<1x96x56x56xf32>
    %326 = linalg.generic {indexing_maps = [#map5, #map1, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%279, %325 : tensor<f32>, tensor<1x96x56x56xf32>) outs(%315 : tensor<1x96x56x56xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.cmpf olt, %in, %in_246 : f32
      %1281 = arith.select %1280, %in, %in_246 : f32
      linalg.yield %1281 : f32
    } -> tensor<1x96x56x56xf32>
    %327 = tensor.empty() : tensor<1x24x56x56xf32>
    %328 = linalg.fill ins(%cst_1 : f32) outs(%327 : tensor<1x24x56x56xf32>) -> tensor<1x24x56x56xf32>
    %329 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%326, %26 : tensor<1x96x56x56xf32>, tensor<24x96x1x1xf32>) outs(%328 : tensor<1x24x56x56xf32>) -> tensor<1x24x56x56xf32>
    %330 = tensor.empty() : tensor<24xf32>
    %331 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%28 : tensor<24xf32>) outs(%330 : tensor<24xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = arith.truncf %cst_3 : f64 to f32
      %1281 = arith.addf %in, %1280 : f32
      linalg.yield %1281 : f32
    } -> tensor<24xf32>
    %332 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%331 : tensor<24xf32>) outs(%330 : tensor<24xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = math.sqrt %in : f32
      linalg.yield %1280 : f32
    } -> tensor<24xf32>
    %333 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%332 : tensor<24xf32>) outs(%330 : tensor<24xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = arith.cmpf one, %in, %cst_1 : f32
      cf.assert %1280, "unimplemented: tensor with zero element"
      %1281 = arith.divf %cst_2, %in : f32
      linalg.yield %1281 : f32
    } -> tensor<24xf32>
    %expanded_27 = tensor.expand_shape %27 [[0, 1, 2]] output_shape [24, 1, 1] : tensor<24xf32> into tensor<24x1x1xf32>
    %expanded_28 = tensor.expand_shape %333 [[0, 1, 2]] output_shape [24, 1, 1] : tensor<24xf32> into tensor<24x1x1xf32>
    %334 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%329, %expanded_27 : tensor<1x24x56x56xf32>, tensor<24x1x1xf32>) outs(%327 : tensor<1x24x56x56xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.subf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x24x56x56xf32>
    %335 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%334, %expanded_28 : tensor<1x24x56x56xf32>, tensor<24x1x1xf32>) outs(%327 : tensor<1x24x56x56xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.mulf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x24x56x56xf32>
    %expanded_29 = tensor.expand_shape %29 [[0, 1, 2]] output_shape [24, 1, 1] : tensor<24xf32> into tensor<24x1x1xf32>
    %336 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%335, %expanded_29 : tensor<1x24x56x56xf32>, tensor<24x1x1xf32>) outs(%327 : tensor<1x24x56x56xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.mulf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x24x56x56xf32>
    %expanded_30 = tensor.expand_shape %30 [[0, 1, 2]] output_shape [24, 1, 1] : tensor<24xf32> into tensor<24x1x1xf32>
    %337 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%336, %expanded_30 : tensor<1x24x56x56xf32>, tensor<24x1x1xf32>) outs(%327 : tensor<1x24x56x56xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.addf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x24x56x56xf32>
    %338 = tensor.empty() : tensor<1x144x56x56xf32>
    %339 = linalg.fill ins(%cst_1 : f32) outs(%338 : tensor<1x144x56x56xf32>) -> tensor<1x144x56x56xf32>
    %340 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%337, %31 : tensor<1x24x56x56xf32>, tensor<144x24x1x1xf32>) outs(%339 : tensor<1x144x56x56xf32>) -> tensor<1x144x56x56xf32>
    %341 = tensor.empty() : tensor<144xf32>
    %342 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%33 : tensor<144xf32>) outs(%341 : tensor<144xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = arith.truncf %cst_3 : f64 to f32
      %1281 = arith.addf %in, %1280 : f32
      linalg.yield %1281 : f32
    } -> tensor<144xf32>
    %343 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%342 : tensor<144xf32>) outs(%341 : tensor<144xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = math.sqrt %in : f32
      linalg.yield %1280 : f32
    } -> tensor<144xf32>
    %344 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%343 : tensor<144xf32>) outs(%341 : tensor<144xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = arith.cmpf one, %in, %cst_1 : f32
      cf.assert %1280, "unimplemented: tensor with zero element"
      %1281 = arith.divf %cst_2, %in : f32
      linalg.yield %1281 : f32
    } -> tensor<144xf32>
    %expanded_31 = tensor.expand_shape %32 [[0, 1, 2]] output_shape [144, 1, 1] : tensor<144xf32> into tensor<144x1x1xf32>
    %expanded_32 = tensor.expand_shape %344 [[0, 1, 2]] output_shape [144, 1, 1] : tensor<144xf32> into tensor<144x1x1xf32>
    %345 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%340, %expanded_31 : tensor<1x144x56x56xf32>, tensor<144x1x1xf32>) outs(%338 : tensor<1x144x56x56xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.subf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x144x56x56xf32>
    %346 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%345, %expanded_32 : tensor<1x144x56x56xf32>, tensor<144x1x1xf32>) outs(%338 : tensor<1x144x56x56xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.mulf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x144x56x56xf32>
    %expanded_33 = tensor.expand_shape %34 [[0, 1, 2]] output_shape [144, 1, 1] : tensor<144xf32> into tensor<144x1x1xf32>
    %347 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%346, %expanded_33 : tensor<1x144x56x56xf32>, tensor<144x1x1xf32>) outs(%338 : tensor<1x144x56x56xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.mulf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x144x56x56xf32>
    %expanded_34 = tensor.expand_shape %35 [[0, 1, 2]] output_shape [144, 1, 1] : tensor<144xf32> into tensor<144x1x1xf32>
    %348 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%347, %expanded_34 : tensor<1x144x56x56xf32>, tensor<144x1x1xf32>) outs(%338 : tensor<1x144x56x56xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.addf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x144x56x56xf32>
    %349 = linalg.generic {indexing_maps = [#map1, #map5, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%348, %277 : tensor<1x144x56x56xf32>, tensor<f32>) outs(%338 : tensor<1x144x56x56xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.cmpf ogt, %in, %in_246 : f32
      %1281 = arith.select %1280, %in, %in_246 : f32
      linalg.yield %1281 : f32
    } -> tensor<1x144x56x56xf32>
    %350 = linalg.generic {indexing_maps = [#map5, #map1, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%279, %349 : tensor<f32>, tensor<1x144x56x56xf32>) outs(%338 : tensor<1x144x56x56xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.cmpf olt, %in, %in_246 : f32
      %1281 = arith.select %1280, %in, %in_246 : f32
      linalg.yield %1281 : f32
    } -> tensor<1x144x56x56xf32>
    %padded_35 = tensor.pad %350 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg265: index, %arg266: index, %arg267: index, %arg268: index):
      tensor.yield %cst_1 : f32
    } : tensor<1x144x56x56xf32> to tensor<1x144x58x58xf32>
    %collapsed_36 = tensor.collapse_shape %36 [[0, 1], [2], [3]] : tensor<144x1x3x3xf32> into tensor<144x3x3xf32>
    %351 = linalg.depthwise_conv_2d_nchw_chw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_35, %collapsed_36 : tensor<1x144x58x58xf32>, tensor<144x3x3xf32>) outs(%339 : tensor<1x144x56x56xf32>) -> tensor<1x144x56x56xf32>
    %352 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%38 : tensor<144xf32>) outs(%341 : tensor<144xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = arith.truncf %cst_3 : f64 to f32
      %1281 = arith.addf %in, %1280 : f32
      linalg.yield %1281 : f32
    } -> tensor<144xf32>
    %353 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%352 : tensor<144xf32>) outs(%341 : tensor<144xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = math.sqrt %in : f32
      linalg.yield %1280 : f32
    } -> tensor<144xf32>
    %354 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%353 : tensor<144xf32>) outs(%341 : tensor<144xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = arith.cmpf one, %in, %cst_1 : f32
      cf.assert %1280, "unimplemented: tensor with zero element"
      %1281 = arith.divf %cst_2, %in : f32
      linalg.yield %1281 : f32
    } -> tensor<144xf32>
    %expanded_37 = tensor.expand_shape %37 [[0, 1, 2]] output_shape [144, 1, 1] : tensor<144xf32> into tensor<144x1x1xf32>
    %expanded_38 = tensor.expand_shape %354 [[0, 1, 2]] output_shape [144, 1, 1] : tensor<144xf32> into tensor<144x1x1xf32>
    %355 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%351, %expanded_37 : tensor<1x144x56x56xf32>, tensor<144x1x1xf32>) outs(%338 : tensor<1x144x56x56xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.subf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x144x56x56xf32>
    %356 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%355, %expanded_38 : tensor<1x144x56x56xf32>, tensor<144x1x1xf32>) outs(%338 : tensor<1x144x56x56xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.mulf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x144x56x56xf32>
    %expanded_39 = tensor.expand_shape %39 [[0, 1, 2]] output_shape [144, 1, 1] : tensor<144xf32> into tensor<144x1x1xf32>
    %357 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%356, %expanded_39 : tensor<1x144x56x56xf32>, tensor<144x1x1xf32>) outs(%338 : tensor<1x144x56x56xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.mulf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x144x56x56xf32>
    %expanded_40 = tensor.expand_shape %40 [[0, 1, 2]] output_shape [144, 1, 1] : tensor<144xf32> into tensor<144x1x1xf32>
    %358 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%357, %expanded_40 : tensor<1x144x56x56xf32>, tensor<144x1x1xf32>) outs(%338 : tensor<1x144x56x56xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.addf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x144x56x56xf32>
    %359 = linalg.generic {indexing_maps = [#map1, #map5, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%358, %277 : tensor<1x144x56x56xf32>, tensor<f32>) outs(%338 : tensor<1x144x56x56xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.cmpf ogt, %in, %in_246 : f32
      %1281 = arith.select %1280, %in, %in_246 : f32
      linalg.yield %1281 : f32
    } -> tensor<1x144x56x56xf32>
    %360 = linalg.generic {indexing_maps = [#map5, #map1, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%279, %359 : tensor<f32>, tensor<1x144x56x56xf32>) outs(%338 : tensor<1x144x56x56xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.cmpf olt, %in, %in_246 : f32
      %1281 = arith.select %1280, %in, %in_246 : f32
      linalg.yield %1281 : f32
    } -> tensor<1x144x56x56xf32>
    %361 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%360, %41 : tensor<1x144x56x56xf32>, tensor<24x144x1x1xf32>) outs(%328 : tensor<1x24x56x56xf32>) -> tensor<1x24x56x56xf32>
    %362 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%43 : tensor<24xf32>) outs(%330 : tensor<24xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = arith.truncf %cst_3 : f64 to f32
      %1281 = arith.addf %in, %1280 : f32
      linalg.yield %1281 : f32
    } -> tensor<24xf32>
    %363 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%362 : tensor<24xf32>) outs(%330 : tensor<24xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = math.sqrt %in : f32
      linalg.yield %1280 : f32
    } -> tensor<24xf32>
    %364 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%363 : tensor<24xf32>) outs(%330 : tensor<24xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = arith.cmpf one, %in, %cst_1 : f32
      cf.assert %1280, "unimplemented: tensor with zero element"
      %1281 = arith.divf %cst_2, %in : f32
      linalg.yield %1281 : f32
    } -> tensor<24xf32>
    %expanded_41 = tensor.expand_shape %42 [[0, 1, 2]] output_shape [24, 1, 1] : tensor<24xf32> into tensor<24x1x1xf32>
    %expanded_42 = tensor.expand_shape %364 [[0, 1, 2]] output_shape [24, 1, 1] : tensor<24xf32> into tensor<24x1x1xf32>
    %365 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%361, %expanded_41 : tensor<1x24x56x56xf32>, tensor<24x1x1xf32>) outs(%327 : tensor<1x24x56x56xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.subf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x24x56x56xf32>
    %366 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%365, %expanded_42 : tensor<1x24x56x56xf32>, tensor<24x1x1xf32>) outs(%327 : tensor<1x24x56x56xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.mulf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x24x56x56xf32>
    %expanded_43 = tensor.expand_shape %44 [[0, 1, 2]] output_shape [24, 1, 1] : tensor<24xf32> into tensor<24x1x1xf32>
    %367 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%366, %expanded_43 : tensor<1x24x56x56xf32>, tensor<24x1x1xf32>) outs(%327 : tensor<1x24x56x56xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.mulf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x24x56x56xf32>
    %expanded_44 = tensor.expand_shape %45 [[0, 1, 2]] output_shape [24, 1, 1] : tensor<24xf32> into tensor<24x1x1xf32>
    %368 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%367, %expanded_44 : tensor<1x24x56x56xf32>, tensor<24x1x1xf32>) outs(%327 : tensor<1x24x56x56xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.addf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x24x56x56xf32>
    %369 = linalg.generic {indexing_maps = [#map1, #map1, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%337, %368 : tensor<1x24x56x56xf32>, tensor<1x24x56x56xf32>) outs(%327 : tensor<1x24x56x56xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.addf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x24x56x56xf32>
    %370 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%369, %46 : tensor<1x24x56x56xf32>, tensor<144x24x1x1xf32>) outs(%339 : tensor<1x144x56x56xf32>) -> tensor<1x144x56x56xf32>
    %371 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%48 : tensor<144xf32>) outs(%341 : tensor<144xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = arith.truncf %cst_3 : f64 to f32
      %1281 = arith.addf %in, %1280 : f32
      linalg.yield %1281 : f32
    } -> tensor<144xf32>
    %372 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%371 : tensor<144xf32>) outs(%341 : tensor<144xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = math.sqrt %in : f32
      linalg.yield %1280 : f32
    } -> tensor<144xf32>
    %373 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%372 : tensor<144xf32>) outs(%341 : tensor<144xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = arith.cmpf one, %in, %cst_1 : f32
      cf.assert %1280, "unimplemented: tensor with zero element"
      %1281 = arith.divf %cst_2, %in : f32
      linalg.yield %1281 : f32
    } -> tensor<144xf32>
    %expanded_45 = tensor.expand_shape %47 [[0, 1, 2]] output_shape [144, 1, 1] : tensor<144xf32> into tensor<144x1x1xf32>
    %expanded_46 = tensor.expand_shape %373 [[0, 1, 2]] output_shape [144, 1, 1] : tensor<144xf32> into tensor<144x1x1xf32>
    %374 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%370, %expanded_45 : tensor<1x144x56x56xf32>, tensor<144x1x1xf32>) outs(%338 : tensor<1x144x56x56xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.subf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x144x56x56xf32>
    %375 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%374, %expanded_46 : tensor<1x144x56x56xf32>, tensor<144x1x1xf32>) outs(%338 : tensor<1x144x56x56xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.mulf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x144x56x56xf32>
    %expanded_47 = tensor.expand_shape %49 [[0, 1, 2]] output_shape [144, 1, 1] : tensor<144xf32> into tensor<144x1x1xf32>
    %376 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%375, %expanded_47 : tensor<1x144x56x56xf32>, tensor<144x1x1xf32>) outs(%338 : tensor<1x144x56x56xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.mulf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x144x56x56xf32>
    %expanded_48 = tensor.expand_shape %50 [[0, 1, 2]] output_shape [144, 1, 1] : tensor<144xf32> into tensor<144x1x1xf32>
    %377 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%376, %expanded_48 : tensor<1x144x56x56xf32>, tensor<144x1x1xf32>) outs(%338 : tensor<1x144x56x56xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.addf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x144x56x56xf32>
    %378 = linalg.generic {indexing_maps = [#map1, #map5, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%377, %277 : tensor<1x144x56x56xf32>, tensor<f32>) outs(%338 : tensor<1x144x56x56xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.cmpf ogt, %in, %in_246 : f32
      %1281 = arith.select %1280, %in, %in_246 : f32
      linalg.yield %1281 : f32
    } -> tensor<1x144x56x56xf32>
    %379 = linalg.generic {indexing_maps = [#map5, #map1, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%279, %378 : tensor<f32>, tensor<1x144x56x56xf32>) outs(%338 : tensor<1x144x56x56xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.cmpf olt, %in, %in_246 : f32
      %1281 = arith.select %1280, %in, %in_246 : f32
      linalg.yield %1281 : f32
    } -> tensor<1x144x56x56xf32>
    %padded_49 = tensor.pad %379 low[0, 0, 0, 0] high[0, 0, 1, 1] {
    ^bb0(%arg265: index, %arg266: index, %arg267: index, %arg268: index):
      tensor.yield %cst_1 : f32
    } : tensor<1x144x56x56xf32> to tensor<1x144x57x57xf32>
    %380 = tensor.empty() : tensor<1x144x28x28xf32>
    %381 = linalg.fill ins(%cst_1 : f32) outs(%380 : tensor<1x144x28x28xf32>) -> tensor<1x144x28x28xf32>
    %collapsed_50 = tensor.collapse_shape %51 [[0, 1], [2], [3]] : tensor<144x1x3x3xf32> into tensor<144x3x3xf32>
    %382 = linalg.depthwise_conv_2d_nchw_chw {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%padded_49, %collapsed_50 : tensor<1x144x57x57xf32>, tensor<144x3x3xf32>) outs(%381 : tensor<1x144x28x28xf32>) -> tensor<1x144x28x28xf32>
    %383 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%53 : tensor<144xf32>) outs(%341 : tensor<144xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = arith.truncf %cst_3 : f64 to f32
      %1281 = arith.addf %in, %1280 : f32
      linalg.yield %1281 : f32
    } -> tensor<144xf32>
    %384 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%383 : tensor<144xf32>) outs(%341 : tensor<144xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = math.sqrt %in : f32
      linalg.yield %1280 : f32
    } -> tensor<144xf32>
    %385 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%384 : tensor<144xf32>) outs(%341 : tensor<144xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = arith.cmpf one, %in, %cst_1 : f32
      cf.assert %1280, "unimplemented: tensor with zero element"
      %1281 = arith.divf %cst_2, %in : f32
      linalg.yield %1281 : f32
    } -> tensor<144xf32>
    %expanded_51 = tensor.expand_shape %52 [[0, 1, 2]] output_shape [144, 1, 1] : tensor<144xf32> into tensor<144x1x1xf32>
    %expanded_52 = tensor.expand_shape %385 [[0, 1, 2]] output_shape [144, 1, 1] : tensor<144xf32> into tensor<144x1x1xf32>
    %386 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%382, %expanded_51 : tensor<1x144x28x28xf32>, tensor<144x1x1xf32>) outs(%380 : tensor<1x144x28x28xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.subf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x144x28x28xf32>
    %387 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%386, %expanded_52 : tensor<1x144x28x28xf32>, tensor<144x1x1xf32>) outs(%380 : tensor<1x144x28x28xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.mulf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x144x28x28xf32>
    %expanded_53 = tensor.expand_shape %54 [[0, 1, 2]] output_shape [144, 1, 1] : tensor<144xf32> into tensor<144x1x1xf32>
    %388 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%387, %expanded_53 : tensor<1x144x28x28xf32>, tensor<144x1x1xf32>) outs(%380 : tensor<1x144x28x28xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.mulf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x144x28x28xf32>
    %expanded_54 = tensor.expand_shape %55 [[0, 1, 2]] output_shape [144, 1, 1] : tensor<144xf32> into tensor<144x1x1xf32>
    %389 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%388, %expanded_54 : tensor<1x144x28x28xf32>, tensor<144x1x1xf32>) outs(%380 : tensor<1x144x28x28xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.addf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x144x28x28xf32>
    %390 = linalg.generic {indexing_maps = [#map1, #map5, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%389, %277 : tensor<1x144x28x28xf32>, tensor<f32>) outs(%380 : tensor<1x144x28x28xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.cmpf ogt, %in, %in_246 : f32
      %1281 = arith.select %1280, %in, %in_246 : f32
      linalg.yield %1281 : f32
    } -> tensor<1x144x28x28xf32>
    %391 = linalg.generic {indexing_maps = [#map5, #map1, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%279, %390 : tensor<f32>, tensor<1x144x28x28xf32>) outs(%380 : tensor<1x144x28x28xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.cmpf olt, %in, %in_246 : f32
      %1281 = arith.select %1280, %in, %in_246 : f32
      linalg.yield %1281 : f32
    } -> tensor<1x144x28x28xf32>
    %392 = tensor.empty() : tensor<1x32x28x28xf32>
    %393 = linalg.fill ins(%cst_1 : f32) outs(%392 : tensor<1x32x28x28xf32>) -> tensor<1x32x28x28xf32>
    %394 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%391, %56 : tensor<1x144x28x28xf32>, tensor<32x144x1x1xf32>) outs(%393 : tensor<1x32x28x28xf32>) -> tensor<1x32x28x28xf32>
    %395 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%58 : tensor<32xf32>) outs(%266 : tensor<32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = arith.truncf %cst_3 : f64 to f32
      %1281 = arith.addf %in, %1280 : f32
      linalg.yield %1281 : f32
    } -> tensor<32xf32>
    %396 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%395 : tensor<32xf32>) outs(%266 : tensor<32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = math.sqrt %in : f32
      linalg.yield %1280 : f32
    } -> tensor<32xf32>
    %397 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%396 : tensor<32xf32>) outs(%266 : tensor<32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = arith.cmpf one, %in, %cst_1 : f32
      cf.assert %1280, "unimplemented: tensor with zero element"
      %1281 = arith.divf %cst_2, %in : f32
      linalg.yield %1281 : f32
    } -> tensor<32xf32>
    %expanded_55 = tensor.expand_shape %57 [[0, 1, 2]] output_shape [32, 1, 1] : tensor<32xf32> into tensor<32x1x1xf32>
    %expanded_56 = tensor.expand_shape %397 [[0, 1, 2]] output_shape [32, 1, 1] : tensor<32xf32> into tensor<32x1x1xf32>
    %398 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%394, %expanded_55 : tensor<1x32x28x28xf32>, tensor<32x1x1xf32>) outs(%392 : tensor<1x32x28x28xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.subf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x32x28x28xf32>
    %399 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%398, %expanded_56 : tensor<1x32x28x28xf32>, tensor<32x1x1xf32>) outs(%392 : tensor<1x32x28x28xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.mulf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x32x28x28xf32>
    %expanded_57 = tensor.expand_shape %59 [[0, 1, 2]] output_shape [32, 1, 1] : tensor<32xf32> into tensor<32x1x1xf32>
    %400 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%399, %expanded_57 : tensor<1x32x28x28xf32>, tensor<32x1x1xf32>) outs(%392 : tensor<1x32x28x28xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.mulf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x32x28x28xf32>
    %expanded_58 = tensor.expand_shape %60 [[0, 1, 2]] output_shape [32, 1, 1] : tensor<32xf32> into tensor<32x1x1xf32>
    %401 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%400, %expanded_58 : tensor<1x32x28x28xf32>, tensor<32x1x1xf32>) outs(%392 : tensor<1x32x28x28xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.addf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x32x28x28xf32>
    %402 = tensor.empty() : tensor<1x192x28x28xf32>
    %403 = linalg.fill ins(%cst_1 : f32) outs(%402 : tensor<1x192x28x28xf32>) -> tensor<1x192x28x28xf32>
    %404 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%401, %61 : tensor<1x32x28x28xf32>, tensor<192x32x1x1xf32>) outs(%403 : tensor<1x192x28x28xf32>) -> tensor<1x192x28x28xf32>
    %405 = tensor.empty() : tensor<192xf32>
    %406 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%63 : tensor<192xf32>) outs(%405 : tensor<192xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = arith.truncf %cst_3 : f64 to f32
      %1281 = arith.addf %in, %1280 : f32
      linalg.yield %1281 : f32
    } -> tensor<192xf32>
    %407 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%406 : tensor<192xf32>) outs(%405 : tensor<192xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = math.sqrt %in : f32
      linalg.yield %1280 : f32
    } -> tensor<192xf32>
    %408 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%407 : tensor<192xf32>) outs(%405 : tensor<192xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = arith.cmpf one, %in, %cst_1 : f32
      cf.assert %1280, "unimplemented: tensor with zero element"
      %1281 = arith.divf %cst_2, %in : f32
      linalg.yield %1281 : f32
    } -> tensor<192xf32>
    %expanded_59 = tensor.expand_shape %62 [[0, 1, 2]] output_shape [192, 1, 1] : tensor<192xf32> into tensor<192x1x1xf32>
    %expanded_60 = tensor.expand_shape %408 [[0, 1, 2]] output_shape [192, 1, 1] : tensor<192xf32> into tensor<192x1x1xf32>
    %409 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%404, %expanded_59 : tensor<1x192x28x28xf32>, tensor<192x1x1xf32>) outs(%402 : tensor<1x192x28x28xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.subf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x192x28x28xf32>
    %410 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%409, %expanded_60 : tensor<1x192x28x28xf32>, tensor<192x1x1xf32>) outs(%402 : tensor<1x192x28x28xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.mulf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x192x28x28xf32>
    %expanded_61 = tensor.expand_shape %64 [[0, 1, 2]] output_shape [192, 1, 1] : tensor<192xf32> into tensor<192x1x1xf32>
    %411 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%410, %expanded_61 : tensor<1x192x28x28xf32>, tensor<192x1x1xf32>) outs(%402 : tensor<1x192x28x28xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.mulf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x192x28x28xf32>
    %expanded_62 = tensor.expand_shape %65 [[0, 1, 2]] output_shape [192, 1, 1] : tensor<192xf32> into tensor<192x1x1xf32>
    %412 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%411, %expanded_62 : tensor<1x192x28x28xf32>, tensor<192x1x1xf32>) outs(%402 : tensor<1x192x28x28xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.addf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x192x28x28xf32>
    %413 = linalg.generic {indexing_maps = [#map1, #map5, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%412, %277 : tensor<1x192x28x28xf32>, tensor<f32>) outs(%402 : tensor<1x192x28x28xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.cmpf ogt, %in, %in_246 : f32
      %1281 = arith.select %1280, %in, %in_246 : f32
      linalg.yield %1281 : f32
    } -> tensor<1x192x28x28xf32>
    %414 = linalg.generic {indexing_maps = [#map5, #map1, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%279, %413 : tensor<f32>, tensor<1x192x28x28xf32>) outs(%402 : tensor<1x192x28x28xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.cmpf olt, %in, %in_246 : f32
      %1281 = arith.select %1280, %in, %in_246 : f32
      linalg.yield %1281 : f32
    } -> tensor<1x192x28x28xf32>
    %padded_63 = tensor.pad %414 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg265: index, %arg266: index, %arg267: index, %arg268: index):
      tensor.yield %cst_1 : f32
    } : tensor<1x192x28x28xf32> to tensor<1x192x30x30xf32>
    %collapsed_64 = tensor.collapse_shape %66 [[0, 1], [2], [3]] : tensor<192x1x3x3xf32> into tensor<192x3x3xf32>
    %415 = linalg.depthwise_conv_2d_nchw_chw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_63, %collapsed_64 : tensor<1x192x30x30xf32>, tensor<192x3x3xf32>) outs(%403 : tensor<1x192x28x28xf32>) -> tensor<1x192x28x28xf32>
    %416 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%68 : tensor<192xf32>) outs(%405 : tensor<192xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = arith.truncf %cst_3 : f64 to f32
      %1281 = arith.addf %in, %1280 : f32
      linalg.yield %1281 : f32
    } -> tensor<192xf32>
    %417 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%416 : tensor<192xf32>) outs(%405 : tensor<192xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = math.sqrt %in : f32
      linalg.yield %1280 : f32
    } -> tensor<192xf32>
    %418 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%417 : tensor<192xf32>) outs(%405 : tensor<192xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = arith.cmpf one, %in, %cst_1 : f32
      cf.assert %1280, "unimplemented: tensor with zero element"
      %1281 = arith.divf %cst_2, %in : f32
      linalg.yield %1281 : f32
    } -> tensor<192xf32>
    %expanded_65 = tensor.expand_shape %67 [[0, 1, 2]] output_shape [192, 1, 1] : tensor<192xf32> into tensor<192x1x1xf32>
    %expanded_66 = tensor.expand_shape %418 [[0, 1, 2]] output_shape [192, 1, 1] : tensor<192xf32> into tensor<192x1x1xf32>
    %419 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%415, %expanded_65 : tensor<1x192x28x28xf32>, tensor<192x1x1xf32>) outs(%402 : tensor<1x192x28x28xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.subf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x192x28x28xf32>
    %420 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%419, %expanded_66 : tensor<1x192x28x28xf32>, tensor<192x1x1xf32>) outs(%402 : tensor<1x192x28x28xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.mulf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x192x28x28xf32>
    %expanded_67 = tensor.expand_shape %69 [[0, 1, 2]] output_shape [192, 1, 1] : tensor<192xf32> into tensor<192x1x1xf32>
    %421 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%420, %expanded_67 : tensor<1x192x28x28xf32>, tensor<192x1x1xf32>) outs(%402 : tensor<1x192x28x28xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.mulf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x192x28x28xf32>
    %expanded_68 = tensor.expand_shape %70 [[0, 1, 2]] output_shape [192, 1, 1] : tensor<192xf32> into tensor<192x1x1xf32>
    %422 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%421, %expanded_68 : tensor<1x192x28x28xf32>, tensor<192x1x1xf32>) outs(%402 : tensor<1x192x28x28xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.addf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x192x28x28xf32>
    %423 = linalg.generic {indexing_maps = [#map1, #map5, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%422, %277 : tensor<1x192x28x28xf32>, tensor<f32>) outs(%402 : tensor<1x192x28x28xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.cmpf ogt, %in, %in_246 : f32
      %1281 = arith.select %1280, %in, %in_246 : f32
      linalg.yield %1281 : f32
    } -> tensor<1x192x28x28xf32>
    %424 = linalg.generic {indexing_maps = [#map5, #map1, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%279, %423 : tensor<f32>, tensor<1x192x28x28xf32>) outs(%402 : tensor<1x192x28x28xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.cmpf olt, %in, %in_246 : f32
      %1281 = arith.select %1280, %in, %in_246 : f32
      linalg.yield %1281 : f32
    } -> tensor<1x192x28x28xf32>
    %425 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%424, %71 : tensor<1x192x28x28xf32>, tensor<32x192x1x1xf32>) outs(%393 : tensor<1x32x28x28xf32>) -> tensor<1x32x28x28xf32>
    %426 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%73 : tensor<32xf32>) outs(%266 : tensor<32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = arith.truncf %cst_3 : f64 to f32
      %1281 = arith.addf %in, %1280 : f32
      linalg.yield %1281 : f32
    } -> tensor<32xf32>
    %427 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%426 : tensor<32xf32>) outs(%266 : tensor<32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = math.sqrt %in : f32
      linalg.yield %1280 : f32
    } -> tensor<32xf32>
    %428 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%427 : tensor<32xf32>) outs(%266 : tensor<32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = arith.cmpf one, %in, %cst_1 : f32
      cf.assert %1280, "unimplemented: tensor with zero element"
      %1281 = arith.divf %cst_2, %in : f32
      linalg.yield %1281 : f32
    } -> tensor<32xf32>
    %expanded_69 = tensor.expand_shape %72 [[0, 1, 2]] output_shape [32, 1, 1] : tensor<32xf32> into tensor<32x1x1xf32>
    %expanded_70 = tensor.expand_shape %428 [[0, 1, 2]] output_shape [32, 1, 1] : tensor<32xf32> into tensor<32x1x1xf32>
    %429 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%425, %expanded_69 : tensor<1x32x28x28xf32>, tensor<32x1x1xf32>) outs(%392 : tensor<1x32x28x28xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.subf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x32x28x28xf32>
    %430 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%429, %expanded_70 : tensor<1x32x28x28xf32>, tensor<32x1x1xf32>) outs(%392 : tensor<1x32x28x28xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.mulf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x32x28x28xf32>
    %expanded_71 = tensor.expand_shape %74 [[0, 1, 2]] output_shape [32, 1, 1] : tensor<32xf32> into tensor<32x1x1xf32>
    %431 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%430, %expanded_71 : tensor<1x32x28x28xf32>, tensor<32x1x1xf32>) outs(%392 : tensor<1x32x28x28xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.mulf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x32x28x28xf32>
    %expanded_72 = tensor.expand_shape %75 [[0, 1, 2]] output_shape [32, 1, 1] : tensor<32xf32> into tensor<32x1x1xf32>
    %432 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%431, %expanded_72 : tensor<1x32x28x28xf32>, tensor<32x1x1xf32>) outs(%392 : tensor<1x32x28x28xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.addf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x32x28x28xf32>
    %433 = linalg.generic {indexing_maps = [#map1, #map1, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%401, %432 : tensor<1x32x28x28xf32>, tensor<1x32x28x28xf32>) outs(%392 : tensor<1x32x28x28xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.addf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x32x28x28xf32>
    %434 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%433, %76 : tensor<1x32x28x28xf32>, tensor<192x32x1x1xf32>) outs(%403 : tensor<1x192x28x28xf32>) -> tensor<1x192x28x28xf32>
    %435 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%78 : tensor<192xf32>) outs(%405 : tensor<192xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = arith.truncf %cst_3 : f64 to f32
      %1281 = arith.addf %in, %1280 : f32
      linalg.yield %1281 : f32
    } -> tensor<192xf32>
    %436 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%435 : tensor<192xf32>) outs(%405 : tensor<192xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = math.sqrt %in : f32
      linalg.yield %1280 : f32
    } -> tensor<192xf32>
    %437 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%436 : tensor<192xf32>) outs(%405 : tensor<192xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = arith.cmpf one, %in, %cst_1 : f32
      cf.assert %1280, "unimplemented: tensor with zero element"
      %1281 = arith.divf %cst_2, %in : f32
      linalg.yield %1281 : f32
    } -> tensor<192xf32>
    %expanded_73 = tensor.expand_shape %77 [[0, 1, 2]] output_shape [192, 1, 1] : tensor<192xf32> into tensor<192x1x1xf32>
    %expanded_74 = tensor.expand_shape %437 [[0, 1, 2]] output_shape [192, 1, 1] : tensor<192xf32> into tensor<192x1x1xf32>
    %438 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%434, %expanded_73 : tensor<1x192x28x28xf32>, tensor<192x1x1xf32>) outs(%402 : tensor<1x192x28x28xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.subf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x192x28x28xf32>
    %439 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%438, %expanded_74 : tensor<1x192x28x28xf32>, tensor<192x1x1xf32>) outs(%402 : tensor<1x192x28x28xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.mulf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x192x28x28xf32>
    %expanded_75 = tensor.expand_shape %79 [[0, 1, 2]] output_shape [192, 1, 1] : tensor<192xf32> into tensor<192x1x1xf32>
    %440 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%439, %expanded_75 : tensor<1x192x28x28xf32>, tensor<192x1x1xf32>) outs(%402 : tensor<1x192x28x28xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.mulf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x192x28x28xf32>
    %expanded_76 = tensor.expand_shape %80 [[0, 1, 2]] output_shape [192, 1, 1] : tensor<192xf32> into tensor<192x1x1xf32>
    %441 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%440, %expanded_76 : tensor<1x192x28x28xf32>, tensor<192x1x1xf32>) outs(%402 : tensor<1x192x28x28xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.addf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x192x28x28xf32>
    %442 = linalg.generic {indexing_maps = [#map1, #map5, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%441, %277 : tensor<1x192x28x28xf32>, tensor<f32>) outs(%402 : tensor<1x192x28x28xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.cmpf ogt, %in, %in_246 : f32
      %1281 = arith.select %1280, %in, %in_246 : f32
      linalg.yield %1281 : f32
    } -> tensor<1x192x28x28xf32>
    %443 = linalg.generic {indexing_maps = [#map5, #map1, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%279, %442 : tensor<f32>, tensor<1x192x28x28xf32>) outs(%402 : tensor<1x192x28x28xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.cmpf olt, %in, %in_246 : f32
      %1281 = arith.select %1280, %in, %in_246 : f32
      linalg.yield %1281 : f32
    } -> tensor<1x192x28x28xf32>
    %padded_77 = tensor.pad %443 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg265: index, %arg266: index, %arg267: index, %arg268: index):
      tensor.yield %cst_1 : f32
    } : tensor<1x192x28x28xf32> to tensor<1x192x30x30xf32>
    %collapsed_78 = tensor.collapse_shape %81 [[0, 1], [2], [3]] : tensor<192x1x3x3xf32> into tensor<192x3x3xf32>
    %444 = linalg.depthwise_conv_2d_nchw_chw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_77, %collapsed_78 : tensor<1x192x30x30xf32>, tensor<192x3x3xf32>) outs(%403 : tensor<1x192x28x28xf32>) -> tensor<1x192x28x28xf32>
    %445 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%83 : tensor<192xf32>) outs(%405 : tensor<192xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = arith.truncf %cst_3 : f64 to f32
      %1281 = arith.addf %in, %1280 : f32
      linalg.yield %1281 : f32
    } -> tensor<192xf32>
    %446 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%445 : tensor<192xf32>) outs(%405 : tensor<192xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = math.sqrt %in : f32
      linalg.yield %1280 : f32
    } -> tensor<192xf32>
    %447 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%446 : tensor<192xf32>) outs(%405 : tensor<192xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = arith.cmpf one, %in, %cst_1 : f32
      cf.assert %1280, "unimplemented: tensor with zero element"
      %1281 = arith.divf %cst_2, %in : f32
      linalg.yield %1281 : f32
    } -> tensor<192xf32>
    %expanded_79 = tensor.expand_shape %82 [[0, 1, 2]] output_shape [192, 1, 1] : tensor<192xf32> into tensor<192x1x1xf32>
    %expanded_80 = tensor.expand_shape %447 [[0, 1, 2]] output_shape [192, 1, 1] : tensor<192xf32> into tensor<192x1x1xf32>
    %448 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%444, %expanded_79 : tensor<1x192x28x28xf32>, tensor<192x1x1xf32>) outs(%402 : tensor<1x192x28x28xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.subf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x192x28x28xf32>
    %449 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%448, %expanded_80 : tensor<1x192x28x28xf32>, tensor<192x1x1xf32>) outs(%402 : tensor<1x192x28x28xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.mulf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x192x28x28xf32>
    %expanded_81 = tensor.expand_shape %84 [[0, 1, 2]] output_shape [192, 1, 1] : tensor<192xf32> into tensor<192x1x1xf32>
    %450 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%449, %expanded_81 : tensor<1x192x28x28xf32>, tensor<192x1x1xf32>) outs(%402 : tensor<1x192x28x28xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.mulf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x192x28x28xf32>
    %expanded_82 = tensor.expand_shape %85 [[0, 1, 2]] output_shape [192, 1, 1] : tensor<192xf32> into tensor<192x1x1xf32>
    %451 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%450, %expanded_82 : tensor<1x192x28x28xf32>, tensor<192x1x1xf32>) outs(%402 : tensor<1x192x28x28xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.addf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x192x28x28xf32>
    %452 = linalg.generic {indexing_maps = [#map1, #map5, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%451, %277 : tensor<1x192x28x28xf32>, tensor<f32>) outs(%402 : tensor<1x192x28x28xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.cmpf ogt, %in, %in_246 : f32
      %1281 = arith.select %1280, %in, %in_246 : f32
      linalg.yield %1281 : f32
    } -> tensor<1x192x28x28xf32>
    %453 = linalg.generic {indexing_maps = [#map5, #map1, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%279, %452 : tensor<f32>, tensor<1x192x28x28xf32>) outs(%402 : tensor<1x192x28x28xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.cmpf olt, %in, %in_246 : f32
      %1281 = arith.select %1280, %in, %in_246 : f32
      linalg.yield %1281 : f32
    } -> tensor<1x192x28x28xf32>
    %454 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%453, %86 : tensor<1x192x28x28xf32>, tensor<32x192x1x1xf32>) outs(%393 : tensor<1x32x28x28xf32>) -> tensor<1x32x28x28xf32>
    %455 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%88 : tensor<32xf32>) outs(%266 : tensor<32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = arith.truncf %cst_3 : f64 to f32
      %1281 = arith.addf %in, %1280 : f32
      linalg.yield %1281 : f32
    } -> tensor<32xf32>
    %456 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%455 : tensor<32xf32>) outs(%266 : tensor<32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = math.sqrt %in : f32
      linalg.yield %1280 : f32
    } -> tensor<32xf32>
    %457 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%456 : tensor<32xf32>) outs(%266 : tensor<32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = arith.cmpf one, %in, %cst_1 : f32
      cf.assert %1280, "unimplemented: tensor with zero element"
      %1281 = arith.divf %cst_2, %in : f32
      linalg.yield %1281 : f32
    } -> tensor<32xf32>
    %expanded_83 = tensor.expand_shape %87 [[0, 1, 2]] output_shape [32, 1, 1] : tensor<32xf32> into tensor<32x1x1xf32>
    %expanded_84 = tensor.expand_shape %457 [[0, 1, 2]] output_shape [32, 1, 1] : tensor<32xf32> into tensor<32x1x1xf32>
    %458 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%454, %expanded_83 : tensor<1x32x28x28xf32>, tensor<32x1x1xf32>) outs(%392 : tensor<1x32x28x28xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.subf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x32x28x28xf32>
    %459 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%458, %expanded_84 : tensor<1x32x28x28xf32>, tensor<32x1x1xf32>) outs(%392 : tensor<1x32x28x28xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.mulf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x32x28x28xf32>
    %expanded_85 = tensor.expand_shape %89 [[0, 1, 2]] output_shape [32, 1, 1] : tensor<32xf32> into tensor<32x1x1xf32>
    %460 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%459, %expanded_85 : tensor<1x32x28x28xf32>, tensor<32x1x1xf32>) outs(%392 : tensor<1x32x28x28xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.mulf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x32x28x28xf32>
    %expanded_86 = tensor.expand_shape %90 [[0, 1, 2]] output_shape [32, 1, 1] : tensor<32xf32> into tensor<32x1x1xf32>
    %461 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%460, %expanded_86 : tensor<1x32x28x28xf32>, tensor<32x1x1xf32>) outs(%392 : tensor<1x32x28x28xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.addf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x32x28x28xf32>
    %462 = linalg.generic {indexing_maps = [#map1, #map1, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%433, %461 : tensor<1x32x28x28xf32>, tensor<1x32x28x28xf32>) outs(%392 : tensor<1x32x28x28xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.addf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x32x28x28xf32>
    %463 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%462, %91 : tensor<1x32x28x28xf32>, tensor<192x32x1x1xf32>) outs(%403 : tensor<1x192x28x28xf32>) -> tensor<1x192x28x28xf32>
    %464 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%93 : tensor<192xf32>) outs(%405 : tensor<192xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = arith.truncf %cst_3 : f64 to f32
      %1281 = arith.addf %in, %1280 : f32
      linalg.yield %1281 : f32
    } -> tensor<192xf32>
    %465 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%464 : tensor<192xf32>) outs(%405 : tensor<192xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = math.sqrt %in : f32
      linalg.yield %1280 : f32
    } -> tensor<192xf32>
    %466 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%465 : tensor<192xf32>) outs(%405 : tensor<192xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = arith.cmpf one, %in, %cst_1 : f32
      cf.assert %1280, "unimplemented: tensor with zero element"
      %1281 = arith.divf %cst_2, %in : f32
      linalg.yield %1281 : f32
    } -> tensor<192xf32>
    %expanded_87 = tensor.expand_shape %92 [[0, 1, 2]] output_shape [192, 1, 1] : tensor<192xf32> into tensor<192x1x1xf32>
    %expanded_88 = tensor.expand_shape %466 [[0, 1, 2]] output_shape [192, 1, 1] : tensor<192xf32> into tensor<192x1x1xf32>
    %467 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%463, %expanded_87 : tensor<1x192x28x28xf32>, tensor<192x1x1xf32>) outs(%402 : tensor<1x192x28x28xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.subf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x192x28x28xf32>
    %468 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%467, %expanded_88 : tensor<1x192x28x28xf32>, tensor<192x1x1xf32>) outs(%402 : tensor<1x192x28x28xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.mulf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x192x28x28xf32>
    %expanded_89 = tensor.expand_shape %94 [[0, 1, 2]] output_shape [192, 1, 1] : tensor<192xf32> into tensor<192x1x1xf32>
    %469 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%468, %expanded_89 : tensor<1x192x28x28xf32>, tensor<192x1x1xf32>) outs(%402 : tensor<1x192x28x28xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.mulf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x192x28x28xf32>
    %expanded_90 = tensor.expand_shape %95 [[0, 1, 2]] output_shape [192, 1, 1] : tensor<192xf32> into tensor<192x1x1xf32>
    %470 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%469, %expanded_90 : tensor<1x192x28x28xf32>, tensor<192x1x1xf32>) outs(%402 : tensor<1x192x28x28xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.addf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x192x28x28xf32>
    %471 = linalg.generic {indexing_maps = [#map1, #map5, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%470, %277 : tensor<1x192x28x28xf32>, tensor<f32>) outs(%402 : tensor<1x192x28x28xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.cmpf ogt, %in, %in_246 : f32
      %1281 = arith.select %1280, %in, %in_246 : f32
      linalg.yield %1281 : f32
    } -> tensor<1x192x28x28xf32>
    %472 = linalg.generic {indexing_maps = [#map5, #map1, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%279, %471 : tensor<f32>, tensor<1x192x28x28xf32>) outs(%402 : tensor<1x192x28x28xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.cmpf olt, %in, %in_246 : f32
      %1281 = arith.select %1280, %in, %in_246 : f32
      linalg.yield %1281 : f32
    } -> tensor<1x192x28x28xf32>
    %padded_91 = tensor.pad %472 low[0, 0, 0, 0] high[0, 0, 1, 1] {
    ^bb0(%arg265: index, %arg266: index, %arg267: index, %arg268: index):
      tensor.yield %cst_1 : f32
    } : tensor<1x192x28x28xf32> to tensor<1x192x29x29xf32>
    %473 = tensor.empty() : tensor<1x192x14x14xf32>
    %474 = linalg.fill ins(%cst_1 : f32) outs(%473 : tensor<1x192x14x14xf32>) -> tensor<1x192x14x14xf32>
    %collapsed_92 = tensor.collapse_shape %96 [[0, 1], [2], [3]] : tensor<192x1x3x3xf32> into tensor<192x3x3xf32>
    %475 = linalg.depthwise_conv_2d_nchw_chw {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%padded_91, %collapsed_92 : tensor<1x192x29x29xf32>, tensor<192x3x3xf32>) outs(%474 : tensor<1x192x14x14xf32>) -> tensor<1x192x14x14xf32>
    %476 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%98 : tensor<192xf32>) outs(%405 : tensor<192xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = arith.truncf %cst_3 : f64 to f32
      %1281 = arith.addf %in, %1280 : f32
      linalg.yield %1281 : f32
    } -> tensor<192xf32>
    %477 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%476 : tensor<192xf32>) outs(%405 : tensor<192xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = math.sqrt %in : f32
      linalg.yield %1280 : f32
    } -> tensor<192xf32>
    %478 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%477 : tensor<192xf32>) outs(%405 : tensor<192xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = arith.cmpf one, %in, %cst_1 : f32
      cf.assert %1280, "unimplemented: tensor with zero element"
      %1281 = arith.divf %cst_2, %in : f32
      linalg.yield %1281 : f32
    } -> tensor<192xf32>
    %expanded_93 = tensor.expand_shape %97 [[0, 1, 2]] output_shape [192, 1, 1] : tensor<192xf32> into tensor<192x1x1xf32>
    %expanded_94 = tensor.expand_shape %478 [[0, 1, 2]] output_shape [192, 1, 1] : tensor<192xf32> into tensor<192x1x1xf32>
    %479 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%475, %expanded_93 : tensor<1x192x14x14xf32>, tensor<192x1x1xf32>) outs(%473 : tensor<1x192x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.subf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x192x14x14xf32>
    %480 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%479, %expanded_94 : tensor<1x192x14x14xf32>, tensor<192x1x1xf32>) outs(%473 : tensor<1x192x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.mulf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x192x14x14xf32>
    %expanded_95 = tensor.expand_shape %99 [[0, 1, 2]] output_shape [192, 1, 1] : tensor<192xf32> into tensor<192x1x1xf32>
    %481 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%480, %expanded_95 : tensor<1x192x14x14xf32>, tensor<192x1x1xf32>) outs(%473 : tensor<1x192x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.mulf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x192x14x14xf32>
    %expanded_96 = tensor.expand_shape %100 [[0, 1, 2]] output_shape [192, 1, 1] : tensor<192xf32> into tensor<192x1x1xf32>
    %482 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%481, %expanded_96 : tensor<1x192x14x14xf32>, tensor<192x1x1xf32>) outs(%473 : tensor<1x192x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.addf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x192x14x14xf32>
    %483 = linalg.generic {indexing_maps = [#map1, #map5, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%482, %277 : tensor<1x192x14x14xf32>, tensor<f32>) outs(%473 : tensor<1x192x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.cmpf ogt, %in, %in_246 : f32
      %1281 = arith.select %1280, %in, %in_246 : f32
      linalg.yield %1281 : f32
    } -> tensor<1x192x14x14xf32>
    %484 = linalg.generic {indexing_maps = [#map5, #map1, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%279, %483 : tensor<f32>, tensor<1x192x14x14xf32>) outs(%473 : tensor<1x192x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.cmpf olt, %in, %in_246 : f32
      %1281 = arith.select %1280, %in, %in_246 : f32
      linalg.yield %1281 : f32
    } -> tensor<1x192x14x14xf32>
    %485 = tensor.empty() : tensor<1x64x14x14xf32>
    %486 = linalg.fill ins(%cst_1 : f32) outs(%485 : tensor<1x64x14x14xf32>) -> tensor<1x64x14x14xf32>
    %487 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%484, %101 : tensor<1x192x14x14xf32>, tensor<64x192x1x1xf32>) outs(%486 : tensor<1x64x14x14xf32>) -> tensor<1x64x14x14xf32>
    %488 = tensor.empty() : tensor<64xf32>
    %489 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%103 : tensor<64xf32>) outs(%488 : tensor<64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = arith.truncf %cst_3 : f64 to f32
      %1281 = arith.addf %in, %1280 : f32
      linalg.yield %1281 : f32
    } -> tensor<64xf32>
    %490 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%489 : tensor<64xf32>) outs(%488 : tensor<64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = math.sqrt %in : f32
      linalg.yield %1280 : f32
    } -> tensor<64xf32>
    %491 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%490 : tensor<64xf32>) outs(%488 : tensor<64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = arith.cmpf one, %in, %cst_1 : f32
      cf.assert %1280, "unimplemented: tensor with zero element"
      %1281 = arith.divf %cst_2, %in : f32
      linalg.yield %1281 : f32
    } -> tensor<64xf32>
    %expanded_97 = tensor.expand_shape %102 [[0, 1, 2]] output_shape [64, 1, 1] : tensor<64xf32> into tensor<64x1x1xf32>
    %expanded_98 = tensor.expand_shape %491 [[0, 1, 2]] output_shape [64, 1, 1] : tensor<64xf32> into tensor<64x1x1xf32>
    %492 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%487, %expanded_97 : tensor<1x64x14x14xf32>, tensor<64x1x1xf32>) outs(%485 : tensor<1x64x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.subf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x64x14x14xf32>
    %493 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%492, %expanded_98 : tensor<1x64x14x14xf32>, tensor<64x1x1xf32>) outs(%485 : tensor<1x64x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.mulf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x64x14x14xf32>
    %expanded_99 = tensor.expand_shape %104 [[0, 1, 2]] output_shape [64, 1, 1] : tensor<64xf32> into tensor<64x1x1xf32>
    %494 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%493, %expanded_99 : tensor<1x64x14x14xf32>, tensor<64x1x1xf32>) outs(%485 : tensor<1x64x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.mulf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x64x14x14xf32>
    %expanded_100 = tensor.expand_shape %105 [[0, 1, 2]] output_shape [64, 1, 1] : tensor<64xf32> into tensor<64x1x1xf32>
    %495 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%494, %expanded_100 : tensor<1x64x14x14xf32>, tensor<64x1x1xf32>) outs(%485 : tensor<1x64x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.addf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x64x14x14xf32>
    %496 = tensor.empty() : tensor<1x384x14x14xf32>
    %497 = linalg.fill ins(%cst_1 : f32) outs(%496 : tensor<1x384x14x14xf32>) -> tensor<1x384x14x14xf32>
    %498 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%495, %106 : tensor<1x64x14x14xf32>, tensor<384x64x1x1xf32>) outs(%497 : tensor<1x384x14x14xf32>) -> tensor<1x384x14x14xf32>
    %499 = tensor.empty() : tensor<384xf32>
    %500 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%108 : tensor<384xf32>) outs(%499 : tensor<384xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = arith.truncf %cst_3 : f64 to f32
      %1281 = arith.addf %in, %1280 : f32
      linalg.yield %1281 : f32
    } -> tensor<384xf32>
    %501 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%500 : tensor<384xf32>) outs(%499 : tensor<384xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = math.sqrt %in : f32
      linalg.yield %1280 : f32
    } -> tensor<384xf32>
    %502 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%501 : tensor<384xf32>) outs(%499 : tensor<384xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = arith.cmpf one, %in, %cst_1 : f32
      cf.assert %1280, "unimplemented: tensor with zero element"
      %1281 = arith.divf %cst_2, %in : f32
      linalg.yield %1281 : f32
    } -> tensor<384xf32>
    %expanded_101 = tensor.expand_shape %107 [[0, 1, 2]] output_shape [384, 1, 1] : tensor<384xf32> into tensor<384x1x1xf32>
    %expanded_102 = tensor.expand_shape %502 [[0, 1, 2]] output_shape [384, 1, 1] : tensor<384xf32> into tensor<384x1x1xf32>
    %503 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%498, %expanded_101 : tensor<1x384x14x14xf32>, tensor<384x1x1xf32>) outs(%496 : tensor<1x384x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.subf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x384x14x14xf32>
    %504 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%503, %expanded_102 : tensor<1x384x14x14xf32>, tensor<384x1x1xf32>) outs(%496 : tensor<1x384x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.mulf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x384x14x14xf32>
    %expanded_103 = tensor.expand_shape %109 [[0, 1, 2]] output_shape [384, 1, 1] : tensor<384xf32> into tensor<384x1x1xf32>
    %505 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%504, %expanded_103 : tensor<1x384x14x14xf32>, tensor<384x1x1xf32>) outs(%496 : tensor<1x384x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.mulf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x384x14x14xf32>
    %expanded_104 = tensor.expand_shape %110 [[0, 1, 2]] output_shape [384, 1, 1] : tensor<384xf32> into tensor<384x1x1xf32>
    %506 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%505, %expanded_104 : tensor<1x384x14x14xf32>, tensor<384x1x1xf32>) outs(%496 : tensor<1x384x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.addf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x384x14x14xf32>
    %507 = linalg.generic {indexing_maps = [#map1, #map5, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%506, %277 : tensor<1x384x14x14xf32>, tensor<f32>) outs(%496 : tensor<1x384x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.cmpf ogt, %in, %in_246 : f32
      %1281 = arith.select %1280, %in, %in_246 : f32
      linalg.yield %1281 : f32
    } -> tensor<1x384x14x14xf32>
    %508 = linalg.generic {indexing_maps = [#map5, #map1, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%279, %507 : tensor<f32>, tensor<1x384x14x14xf32>) outs(%496 : tensor<1x384x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.cmpf olt, %in, %in_246 : f32
      %1281 = arith.select %1280, %in, %in_246 : f32
      linalg.yield %1281 : f32
    } -> tensor<1x384x14x14xf32>
    %padded_105 = tensor.pad %508 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg265: index, %arg266: index, %arg267: index, %arg268: index):
      tensor.yield %cst_1 : f32
    } : tensor<1x384x14x14xf32> to tensor<1x384x16x16xf32>
    %collapsed_106 = tensor.collapse_shape %111 [[0, 1], [2], [3]] : tensor<384x1x3x3xf32> into tensor<384x3x3xf32>
    %509 = linalg.depthwise_conv_2d_nchw_chw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_105, %collapsed_106 : tensor<1x384x16x16xf32>, tensor<384x3x3xf32>) outs(%497 : tensor<1x384x14x14xf32>) -> tensor<1x384x14x14xf32>
    %510 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%113 : tensor<384xf32>) outs(%499 : tensor<384xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = arith.truncf %cst_3 : f64 to f32
      %1281 = arith.addf %in, %1280 : f32
      linalg.yield %1281 : f32
    } -> tensor<384xf32>
    %511 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%510 : tensor<384xf32>) outs(%499 : tensor<384xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = math.sqrt %in : f32
      linalg.yield %1280 : f32
    } -> tensor<384xf32>
    %512 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%511 : tensor<384xf32>) outs(%499 : tensor<384xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = arith.cmpf one, %in, %cst_1 : f32
      cf.assert %1280, "unimplemented: tensor with zero element"
      %1281 = arith.divf %cst_2, %in : f32
      linalg.yield %1281 : f32
    } -> tensor<384xf32>
    %expanded_107 = tensor.expand_shape %112 [[0, 1, 2]] output_shape [384, 1, 1] : tensor<384xf32> into tensor<384x1x1xf32>
    %expanded_108 = tensor.expand_shape %512 [[0, 1, 2]] output_shape [384, 1, 1] : tensor<384xf32> into tensor<384x1x1xf32>
    %513 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%509, %expanded_107 : tensor<1x384x14x14xf32>, tensor<384x1x1xf32>) outs(%496 : tensor<1x384x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.subf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x384x14x14xf32>
    %514 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%513, %expanded_108 : tensor<1x384x14x14xf32>, tensor<384x1x1xf32>) outs(%496 : tensor<1x384x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.mulf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x384x14x14xf32>
    %expanded_109 = tensor.expand_shape %114 [[0, 1, 2]] output_shape [384, 1, 1] : tensor<384xf32> into tensor<384x1x1xf32>
    %515 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%514, %expanded_109 : tensor<1x384x14x14xf32>, tensor<384x1x1xf32>) outs(%496 : tensor<1x384x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.mulf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x384x14x14xf32>
    %expanded_110 = tensor.expand_shape %115 [[0, 1, 2]] output_shape [384, 1, 1] : tensor<384xf32> into tensor<384x1x1xf32>
    %516 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%515, %expanded_110 : tensor<1x384x14x14xf32>, tensor<384x1x1xf32>) outs(%496 : tensor<1x384x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.addf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x384x14x14xf32>
    %517 = linalg.generic {indexing_maps = [#map1, #map5, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%516, %277 : tensor<1x384x14x14xf32>, tensor<f32>) outs(%496 : tensor<1x384x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.cmpf ogt, %in, %in_246 : f32
      %1281 = arith.select %1280, %in, %in_246 : f32
      linalg.yield %1281 : f32
    } -> tensor<1x384x14x14xf32>
    %518 = linalg.generic {indexing_maps = [#map5, #map1, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%279, %517 : tensor<f32>, tensor<1x384x14x14xf32>) outs(%496 : tensor<1x384x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.cmpf olt, %in, %in_246 : f32
      %1281 = arith.select %1280, %in, %in_246 : f32
      linalg.yield %1281 : f32
    } -> tensor<1x384x14x14xf32>
    %519 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%518, %116 : tensor<1x384x14x14xf32>, tensor<64x384x1x1xf32>) outs(%486 : tensor<1x64x14x14xf32>) -> tensor<1x64x14x14xf32>
    %520 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%118 : tensor<64xf32>) outs(%488 : tensor<64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = arith.truncf %cst_3 : f64 to f32
      %1281 = arith.addf %in, %1280 : f32
      linalg.yield %1281 : f32
    } -> tensor<64xf32>
    %521 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%520 : tensor<64xf32>) outs(%488 : tensor<64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = math.sqrt %in : f32
      linalg.yield %1280 : f32
    } -> tensor<64xf32>
    %522 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%521 : tensor<64xf32>) outs(%488 : tensor<64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = arith.cmpf one, %in, %cst_1 : f32
      cf.assert %1280, "unimplemented: tensor with zero element"
      %1281 = arith.divf %cst_2, %in : f32
      linalg.yield %1281 : f32
    } -> tensor<64xf32>
    %expanded_111 = tensor.expand_shape %117 [[0, 1, 2]] output_shape [64, 1, 1] : tensor<64xf32> into tensor<64x1x1xf32>
    %expanded_112 = tensor.expand_shape %522 [[0, 1, 2]] output_shape [64, 1, 1] : tensor<64xf32> into tensor<64x1x1xf32>
    %523 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%519, %expanded_111 : tensor<1x64x14x14xf32>, tensor<64x1x1xf32>) outs(%485 : tensor<1x64x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.subf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x64x14x14xf32>
    %524 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%523, %expanded_112 : tensor<1x64x14x14xf32>, tensor<64x1x1xf32>) outs(%485 : tensor<1x64x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.mulf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x64x14x14xf32>
    %expanded_113 = tensor.expand_shape %119 [[0, 1, 2]] output_shape [64, 1, 1] : tensor<64xf32> into tensor<64x1x1xf32>
    %525 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%524, %expanded_113 : tensor<1x64x14x14xf32>, tensor<64x1x1xf32>) outs(%485 : tensor<1x64x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.mulf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x64x14x14xf32>
    %expanded_114 = tensor.expand_shape %120 [[0, 1, 2]] output_shape [64, 1, 1] : tensor<64xf32> into tensor<64x1x1xf32>
    %526 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%525, %expanded_114 : tensor<1x64x14x14xf32>, tensor<64x1x1xf32>) outs(%485 : tensor<1x64x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.addf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x64x14x14xf32>
    %527 = linalg.generic {indexing_maps = [#map1, #map1, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%495, %526 : tensor<1x64x14x14xf32>, tensor<1x64x14x14xf32>) outs(%485 : tensor<1x64x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.addf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x64x14x14xf32>
    %528 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%527, %121 : tensor<1x64x14x14xf32>, tensor<384x64x1x1xf32>) outs(%497 : tensor<1x384x14x14xf32>) -> tensor<1x384x14x14xf32>
    %529 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%123 : tensor<384xf32>) outs(%499 : tensor<384xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = arith.truncf %cst_3 : f64 to f32
      %1281 = arith.addf %in, %1280 : f32
      linalg.yield %1281 : f32
    } -> tensor<384xf32>
    %530 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%529 : tensor<384xf32>) outs(%499 : tensor<384xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = math.sqrt %in : f32
      linalg.yield %1280 : f32
    } -> tensor<384xf32>
    %531 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%530 : tensor<384xf32>) outs(%499 : tensor<384xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = arith.cmpf one, %in, %cst_1 : f32
      cf.assert %1280, "unimplemented: tensor with zero element"
      %1281 = arith.divf %cst_2, %in : f32
      linalg.yield %1281 : f32
    } -> tensor<384xf32>
    %expanded_115 = tensor.expand_shape %122 [[0, 1, 2]] output_shape [384, 1, 1] : tensor<384xf32> into tensor<384x1x1xf32>
    %expanded_116 = tensor.expand_shape %531 [[0, 1, 2]] output_shape [384, 1, 1] : tensor<384xf32> into tensor<384x1x1xf32>
    %532 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%528, %expanded_115 : tensor<1x384x14x14xf32>, tensor<384x1x1xf32>) outs(%496 : tensor<1x384x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.subf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x384x14x14xf32>
    %533 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%532, %expanded_116 : tensor<1x384x14x14xf32>, tensor<384x1x1xf32>) outs(%496 : tensor<1x384x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.mulf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x384x14x14xf32>
    %expanded_117 = tensor.expand_shape %124 [[0, 1, 2]] output_shape [384, 1, 1] : tensor<384xf32> into tensor<384x1x1xf32>
    %534 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%533, %expanded_117 : tensor<1x384x14x14xf32>, tensor<384x1x1xf32>) outs(%496 : tensor<1x384x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.mulf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x384x14x14xf32>
    %expanded_118 = tensor.expand_shape %125 [[0, 1, 2]] output_shape [384, 1, 1] : tensor<384xf32> into tensor<384x1x1xf32>
    %535 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%534, %expanded_118 : tensor<1x384x14x14xf32>, tensor<384x1x1xf32>) outs(%496 : tensor<1x384x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.addf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x384x14x14xf32>
    %536 = linalg.generic {indexing_maps = [#map1, #map5, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%535, %277 : tensor<1x384x14x14xf32>, tensor<f32>) outs(%496 : tensor<1x384x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.cmpf ogt, %in, %in_246 : f32
      %1281 = arith.select %1280, %in, %in_246 : f32
      linalg.yield %1281 : f32
    } -> tensor<1x384x14x14xf32>
    %537 = linalg.generic {indexing_maps = [#map5, #map1, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%279, %536 : tensor<f32>, tensor<1x384x14x14xf32>) outs(%496 : tensor<1x384x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.cmpf olt, %in, %in_246 : f32
      %1281 = arith.select %1280, %in, %in_246 : f32
      linalg.yield %1281 : f32
    } -> tensor<1x384x14x14xf32>
    %padded_119 = tensor.pad %537 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg265: index, %arg266: index, %arg267: index, %arg268: index):
      tensor.yield %cst_1 : f32
    } : tensor<1x384x14x14xf32> to tensor<1x384x16x16xf32>
    %collapsed_120 = tensor.collapse_shape %126 [[0, 1], [2], [3]] : tensor<384x1x3x3xf32> into tensor<384x3x3xf32>
    %538 = linalg.depthwise_conv_2d_nchw_chw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_119, %collapsed_120 : tensor<1x384x16x16xf32>, tensor<384x3x3xf32>) outs(%497 : tensor<1x384x14x14xf32>) -> tensor<1x384x14x14xf32>
    %539 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%128 : tensor<384xf32>) outs(%499 : tensor<384xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = arith.truncf %cst_3 : f64 to f32
      %1281 = arith.addf %in, %1280 : f32
      linalg.yield %1281 : f32
    } -> tensor<384xf32>
    %540 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%539 : tensor<384xf32>) outs(%499 : tensor<384xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = math.sqrt %in : f32
      linalg.yield %1280 : f32
    } -> tensor<384xf32>
    %541 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%540 : tensor<384xf32>) outs(%499 : tensor<384xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = arith.cmpf one, %in, %cst_1 : f32
      cf.assert %1280, "unimplemented: tensor with zero element"
      %1281 = arith.divf %cst_2, %in : f32
      linalg.yield %1281 : f32
    } -> tensor<384xf32>
    %expanded_121 = tensor.expand_shape %127 [[0, 1, 2]] output_shape [384, 1, 1] : tensor<384xf32> into tensor<384x1x1xf32>
    %expanded_122 = tensor.expand_shape %541 [[0, 1, 2]] output_shape [384, 1, 1] : tensor<384xf32> into tensor<384x1x1xf32>
    %542 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%538, %expanded_121 : tensor<1x384x14x14xf32>, tensor<384x1x1xf32>) outs(%496 : tensor<1x384x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.subf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x384x14x14xf32>
    %543 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%542, %expanded_122 : tensor<1x384x14x14xf32>, tensor<384x1x1xf32>) outs(%496 : tensor<1x384x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.mulf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x384x14x14xf32>
    %expanded_123 = tensor.expand_shape %129 [[0, 1, 2]] output_shape [384, 1, 1] : tensor<384xf32> into tensor<384x1x1xf32>
    %544 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%543, %expanded_123 : tensor<1x384x14x14xf32>, tensor<384x1x1xf32>) outs(%496 : tensor<1x384x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.mulf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x384x14x14xf32>
    %expanded_124 = tensor.expand_shape %130 [[0, 1, 2]] output_shape [384, 1, 1] : tensor<384xf32> into tensor<384x1x1xf32>
    %545 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%544, %expanded_124 : tensor<1x384x14x14xf32>, tensor<384x1x1xf32>) outs(%496 : tensor<1x384x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.addf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x384x14x14xf32>
    %546 = linalg.generic {indexing_maps = [#map1, #map5, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%545, %277 : tensor<1x384x14x14xf32>, tensor<f32>) outs(%496 : tensor<1x384x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.cmpf ogt, %in, %in_246 : f32
      %1281 = arith.select %1280, %in, %in_246 : f32
      linalg.yield %1281 : f32
    } -> tensor<1x384x14x14xf32>
    %547 = linalg.generic {indexing_maps = [#map5, #map1, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%279, %546 : tensor<f32>, tensor<1x384x14x14xf32>) outs(%496 : tensor<1x384x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.cmpf olt, %in, %in_246 : f32
      %1281 = arith.select %1280, %in, %in_246 : f32
      linalg.yield %1281 : f32
    } -> tensor<1x384x14x14xf32>
    %548 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%547, %131 : tensor<1x384x14x14xf32>, tensor<64x384x1x1xf32>) outs(%486 : tensor<1x64x14x14xf32>) -> tensor<1x64x14x14xf32>
    %549 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%133 : tensor<64xf32>) outs(%488 : tensor<64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = arith.truncf %cst_3 : f64 to f32
      %1281 = arith.addf %in, %1280 : f32
      linalg.yield %1281 : f32
    } -> tensor<64xf32>
    %550 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%549 : tensor<64xf32>) outs(%488 : tensor<64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = math.sqrt %in : f32
      linalg.yield %1280 : f32
    } -> tensor<64xf32>
    %551 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%550 : tensor<64xf32>) outs(%488 : tensor<64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = arith.cmpf one, %in, %cst_1 : f32
      cf.assert %1280, "unimplemented: tensor with zero element"
      %1281 = arith.divf %cst_2, %in : f32
      linalg.yield %1281 : f32
    } -> tensor<64xf32>
    %expanded_125 = tensor.expand_shape %132 [[0, 1, 2]] output_shape [64, 1, 1] : tensor<64xf32> into tensor<64x1x1xf32>
    %expanded_126 = tensor.expand_shape %551 [[0, 1, 2]] output_shape [64, 1, 1] : tensor<64xf32> into tensor<64x1x1xf32>
    %552 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%548, %expanded_125 : tensor<1x64x14x14xf32>, tensor<64x1x1xf32>) outs(%485 : tensor<1x64x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.subf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x64x14x14xf32>
    %553 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%552, %expanded_126 : tensor<1x64x14x14xf32>, tensor<64x1x1xf32>) outs(%485 : tensor<1x64x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.mulf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x64x14x14xf32>
    %expanded_127 = tensor.expand_shape %134 [[0, 1, 2]] output_shape [64, 1, 1] : tensor<64xf32> into tensor<64x1x1xf32>
    %554 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%553, %expanded_127 : tensor<1x64x14x14xf32>, tensor<64x1x1xf32>) outs(%485 : tensor<1x64x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.mulf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x64x14x14xf32>
    %expanded_128 = tensor.expand_shape %135 [[0, 1, 2]] output_shape [64, 1, 1] : tensor<64xf32> into tensor<64x1x1xf32>
    %555 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%554, %expanded_128 : tensor<1x64x14x14xf32>, tensor<64x1x1xf32>) outs(%485 : tensor<1x64x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.addf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x64x14x14xf32>
    %556 = linalg.generic {indexing_maps = [#map1, #map1, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%527, %555 : tensor<1x64x14x14xf32>, tensor<1x64x14x14xf32>) outs(%485 : tensor<1x64x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.addf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x64x14x14xf32>
    %557 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%556, %136 : tensor<1x64x14x14xf32>, tensor<384x64x1x1xf32>) outs(%497 : tensor<1x384x14x14xf32>) -> tensor<1x384x14x14xf32>
    %558 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%138 : tensor<384xf32>) outs(%499 : tensor<384xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = arith.truncf %cst_3 : f64 to f32
      %1281 = arith.addf %in, %1280 : f32
      linalg.yield %1281 : f32
    } -> tensor<384xf32>
    %559 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%558 : tensor<384xf32>) outs(%499 : tensor<384xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = math.sqrt %in : f32
      linalg.yield %1280 : f32
    } -> tensor<384xf32>
    %560 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%559 : tensor<384xf32>) outs(%499 : tensor<384xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = arith.cmpf one, %in, %cst_1 : f32
      cf.assert %1280, "unimplemented: tensor with zero element"
      %1281 = arith.divf %cst_2, %in : f32
      linalg.yield %1281 : f32
    } -> tensor<384xf32>
    %expanded_129 = tensor.expand_shape %137 [[0, 1, 2]] output_shape [384, 1, 1] : tensor<384xf32> into tensor<384x1x1xf32>
    %expanded_130 = tensor.expand_shape %560 [[0, 1, 2]] output_shape [384, 1, 1] : tensor<384xf32> into tensor<384x1x1xf32>
    %561 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%557, %expanded_129 : tensor<1x384x14x14xf32>, tensor<384x1x1xf32>) outs(%496 : tensor<1x384x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.subf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x384x14x14xf32>
    %562 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%561, %expanded_130 : tensor<1x384x14x14xf32>, tensor<384x1x1xf32>) outs(%496 : tensor<1x384x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.mulf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x384x14x14xf32>
    %expanded_131 = tensor.expand_shape %139 [[0, 1, 2]] output_shape [384, 1, 1] : tensor<384xf32> into tensor<384x1x1xf32>
    %563 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%562, %expanded_131 : tensor<1x384x14x14xf32>, tensor<384x1x1xf32>) outs(%496 : tensor<1x384x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.mulf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x384x14x14xf32>
    %expanded_132 = tensor.expand_shape %140 [[0, 1, 2]] output_shape [384, 1, 1] : tensor<384xf32> into tensor<384x1x1xf32>
    %564 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%563, %expanded_132 : tensor<1x384x14x14xf32>, tensor<384x1x1xf32>) outs(%496 : tensor<1x384x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.addf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x384x14x14xf32>
    %565 = linalg.generic {indexing_maps = [#map1, #map5, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%564, %277 : tensor<1x384x14x14xf32>, tensor<f32>) outs(%496 : tensor<1x384x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.cmpf ogt, %in, %in_246 : f32
      %1281 = arith.select %1280, %in, %in_246 : f32
      linalg.yield %1281 : f32
    } -> tensor<1x384x14x14xf32>
    %566 = linalg.generic {indexing_maps = [#map5, #map1, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%279, %565 : tensor<f32>, tensor<1x384x14x14xf32>) outs(%496 : tensor<1x384x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.cmpf olt, %in, %in_246 : f32
      %1281 = arith.select %1280, %in, %in_246 : f32
      linalg.yield %1281 : f32
    } -> tensor<1x384x14x14xf32>
    %padded_133 = tensor.pad %566 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg265: index, %arg266: index, %arg267: index, %arg268: index):
      tensor.yield %cst_1 : f32
    } : tensor<1x384x14x14xf32> to tensor<1x384x16x16xf32>
    %collapsed_134 = tensor.collapse_shape %141 [[0, 1], [2], [3]] : tensor<384x1x3x3xf32> into tensor<384x3x3xf32>
    %567 = linalg.depthwise_conv_2d_nchw_chw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_133, %collapsed_134 : tensor<1x384x16x16xf32>, tensor<384x3x3xf32>) outs(%497 : tensor<1x384x14x14xf32>) -> tensor<1x384x14x14xf32>
    %568 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%143 : tensor<384xf32>) outs(%499 : tensor<384xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = arith.truncf %cst_3 : f64 to f32
      %1281 = arith.addf %in, %1280 : f32
      linalg.yield %1281 : f32
    } -> tensor<384xf32>
    %569 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%568 : tensor<384xf32>) outs(%499 : tensor<384xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = math.sqrt %in : f32
      linalg.yield %1280 : f32
    } -> tensor<384xf32>
    %570 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%569 : tensor<384xf32>) outs(%499 : tensor<384xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = arith.cmpf one, %in, %cst_1 : f32
      cf.assert %1280, "unimplemented: tensor with zero element"
      %1281 = arith.divf %cst_2, %in : f32
      linalg.yield %1281 : f32
    } -> tensor<384xf32>
    %expanded_135 = tensor.expand_shape %142 [[0, 1, 2]] output_shape [384, 1, 1] : tensor<384xf32> into tensor<384x1x1xf32>
    %expanded_136 = tensor.expand_shape %570 [[0, 1, 2]] output_shape [384, 1, 1] : tensor<384xf32> into tensor<384x1x1xf32>
    %571 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%567, %expanded_135 : tensor<1x384x14x14xf32>, tensor<384x1x1xf32>) outs(%496 : tensor<1x384x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.subf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x384x14x14xf32>
    %572 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%571, %expanded_136 : tensor<1x384x14x14xf32>, tensor<384x1x1xf32>) outs(%496 : tensor<1x384x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.mulf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x384x14x14xf32>
    %expanded_137 = tensor.expand_shape %144 [[0, 1, 2]] output_shape [384, 1, 1] : tensor<384xf32> into tensor<384x1x1xf32>
    %573 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%572, %expanded_137 : tensor<1x384x14x14xf32>, tensor<384x1x1xf32>) outs(%496 : tensor<1x384x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.mulf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x384x14x14xf32>
    %expanded_138 = tensor.expand_shape %145 [[0, 1, 2]] output_shape [384, 1, 1] : tensor<384xf32> into tensor<384x1x1xf32>
    %574 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%573, %expanded_138 : tensor<1x384x14x14xf32>, tensor<384x1x1xf32>) outs(%496 : tensor<1x384x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.addf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x384x14x14xf32>
    %575 = linalg.generic {indexing_maps = [#map1, #map5, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%574, %277 : tensor<1x384x14x14xf32>, tensor<f32>) outs(%496 : tensor<1x384x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.cmpf ogt, %in, %in_246 : f32
      %1281 = arith.select %1280, %in, %in_246 : f32
      linalg.yield %1281 : f32
    } -> tensor<1x384x14x14xf32>
    %576 = linalg.generic {indexing_maps = [#map5, #map1, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%279, %575 : tensor<f32>, tensor<1x384x14x14xf32>) outs(%496 : tensor<1x384x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.cmpf olt, %in, %in_246 : f32
      %1281 = arith.select %1280, %in, %in_246 : f32
      linalg.yield %1281 : f32
    } -> tensor<1x384x14x14xf32>
    %577 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%576, %146 : tensor<1x384x14x14xf32>, tensor<64x384x1x1xf32>) outs(%486 : tensor<1x64x14x14xf32>) -> tensor<1x64x14x14xf32>
    %578 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%148 : tensor<64xf32>) outs(%488 : tensor<64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = arith.truncf %cst_3 : f64 to f32
      %1281 = arith.addf %in, %1280 : f32
      linalg.yield %1281 : f32
    } -> tensor<64xf32>
    %579 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%578 : tensor<64xf32>) outs(%488 : tensor<64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = math.sqrt %in : f32
      linalg.yield %1280 : f32
    } -> tensor<64xf32>
    %580 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%579 : tensor<64xf32>) outs(%488 : tensor<64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = arith.cmpf one, %in, %cst_1 : f32
      cf.assert %1280, "unimplemented: tensor with zero element"
      %1281 = arith.divf %cst_2, %in : f32
      linalg.yield %1281 : f32
    } -> tensor<64xf32>
    %expanded_139 = tensor.expand_shape %147 [[0, 1, 2]] output_shape [64, 1, 1] : tensor<64xf32> into tensor<64x1x1xf32>
    %expanded_140 = tensor.expand_shape %580 [[0, 1, 2]] output_shape [64, 1, 1] : tensor<64xf32> into tensor<64x1x1xf32>
    %581 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%577, %expanded_139 : tensor<1x64x14x14xf32>, tensor<64x1x1xf32>) outs(%485 : tensor<1x64x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.subf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x64x14x14xf32>
    %582 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%581, %expanded_140 : tensor<1x64x14x14xf32>, tensor<64x1x1xf32>) outs(%485 : tensor<1x64x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.mulf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x64x14x14xf32>
    %expanded_141 = tensor.expand_shape %149 [[0, 1, 2]] output_shape [64, 1, 1] : tensor<64xf32> into tensor<64x1x1xf32>
    %583 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%582, %expanded_141 : tensor<1x64x14x14xf32>, tensor<64x1x1xf32>) outs(%485 : tensor<1x64x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.mulf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x64x14x14xf32>
    %expanded_142 = tensor.expand_shape %150 [[0, 1, 2]] output_shape [64, 1, 1] : tensor<64xf32> into tensor<64x1x1xf32>
    %584 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%583, %expanded_142 : tensor<1x64x14x14xf32>, tensor<64x1x1xf32>) outs(%485 : tensor<1x64x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.addf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x64x14x14xf32>
    %585 = linalg.generic {indexing_maps = [#map1, #map1, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%556, %584 : tensor<1x64x14x14xf32>, tensor<1x64x14x14xf32>) outs(%485 : tensor<1x64x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.addf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x64x14x14xf32>
    %586 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%585, %151 : tensor<1x64x14x14xf32>, tensor<384x64x1x1xf32>) outs(%497 : tensor<1x384x14x14xf32>) -> tensor<1x384x14x14xf32>
    %587 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%153 : tensor<384xf32>) outs(%499 : tensor<384xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = arith.truncf %cst_3 : f64 to f32
      %1281 = arith.addf %in, %1280 : f32
      linalg.yield %1281 : f32
    } -> tensor<384xf32>
    %588 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%587 : tensor<384xf32>) outs(%499 : tensor<384xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = math.sqrt %in : f32
      linalg.yield %1280 : f32
    } -> tensor<384xf32>
    %589 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%588 : tensor<384xf32>) outs(%499 : tensor<384xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = arith.cmpf one, %in, %cst_1 : f32
      cf.assert %1280, "unimplemented: tensor with zero element"
      %1281 = arith.divf %cst_2, %in : f32
      linalg.yield %1281 : f32
    } -> tensor<384xf32>
    %expanded_143 = tensor.expand_shape %152 [[0, 1, 2]] output_shape [384, 1, 1] : tensor<384xf32> into tensor<384x1x1xf32>
    %expanded_144 = tensor.expand_shape %589 [[0, 1, 2]] output_shape [384, 1, 1] : tensor<384xf32> into tensor<384x1x1xf32>
    %590 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%586, %expanded_143 : tensor<1x384x14x14xf32>, tensor<384x1x1xf32>) outs(%496 : tensor<1x384x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.subf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x384x14x14xf32>
    %591 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%590, %expanded_144 : tensor<1x384x14x14xf32>, tensor<384x1x1xf32>) outs(%496 : tensor<1x384x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.mulf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x384x14x14xf32>
    %expanded_145 = tensor.expand_shape %154 [[0, 1, 2]] output_shape [384, 1, 1] : tensor<384xf32> into tensor<384x1x1xf32>
    %592 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%591, %expanded_145 : tensor<1x384x14x14xf32>, tensor<384x1x1xf32>) outs(%496 : tensor<1x384x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.mulf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x384x14x14xf32>
    %expanded_146 = tensor.expand_shape %155 [[0, 1, 2]] output_shape [384, 1, 1] : tensor<384xf32> into tensor<384x1x1xf32>
    %593 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%592, %expanded_146 : tensor<1x384x14x14xf32>, tensor<384x1x1xf32>) outs(%496 : tensor<1x384x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.addf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x384x14x14xf32>
    %594 = linalg.generic {indexing_maps = [#map1, #map5, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%593, %277 : tensor<1x384x14x14xf32>, tensor<f32>) outs(%496 : tensor<1x384x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.cmpf ogt, %in, %in_246 : f32
      %1281 = arith.select %1280, %in, %in_246 : f32
      linalg.yield %1281 : f32
    } -> tensor<1x384x14x14xf32>
    %595 = linalg.generic {indexing_maps = [#map5, #map1, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%279, %594 : tensor<f32>, tensor<1x384x14x14xf32>) outs(%496 : tensor<1x384x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.cmpf olt, %in, %in_246 : f32
      %1281 = arith.select %1280, %in, %in_246 : f32
      linalg.yield %1281 : f32
    } -> tensor<1x384x14x14xf32>
    %padded_147 = tensor.pad %595 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg265: index, %arg266: index, %arg267: index, %arg268: index):
      tensor.yield %cst_1 : f32
    } : tensor<1x384x14x14xf32> to tensor<1x384x16x16xf32>
    %collapsed_148 = tensor.collapse_shape %156 [[0, 1], [2], [3]] : tensor<384x1x3x3xf32> into tensor<384x3x3xf32>
    %596 = linalg.depthwise_conv_2d_nchw_chw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_147, %collapsed_148 : tensor<1x384x16x16xf32>, tensor<384x3x3xf32>) outs(%497 : tensor<1x384x14x14xf32>) -> tensor<1x384x14x14xf32>
    %597 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%158 : tensor<384xf32>) outs(%499 : tensor<384xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = arith.truncf %cst_3 : f64 to f32
      %1281 = arith.addf %in, %1280 : f32
      linalg.yield %1281 : f32
    } -> tensor<384xf32>
    %598 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%597 : tensor<384xf32>) outs(%499 : tensor<384xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = math.sqrt %in : f32
      linalg.yield %1280 : f32
    } -> tensor<384xf32>
    %599 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%598 : tensor<384xf32>) outs(%499 : tensor<384xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = arith.cmpf one, %in, %cst_1 : f32
      cf.assert %1280, "unimplemented: tensor with zero element"
      %1281 = arith.divf %cst_2, %in : f32
      linalg.yield %1281 : f32
    } -> tensor<384xf32>
    %expanded_149 = tensor.expand_shape %157 [[0, 1, 2]] output_shape [384, 1, 1] : tensor<384xf32> into tensor<384x1x1xf32>
    %expanded_150 = tensor.expand_shape %599 [[0, 1, 2]] output_shape [384, 1, 1] : tensor<384xf32> into tensor<384x1x1xf32>
    %600 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%596, %expanded_149 : tensor<1x384x14x14xf32>, tensor<384x1x1xf32>) outs(%496 : tensor<1x384x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.subf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x384x14x14xf32>
    %601 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%600, %expanded_150 : tensor<1x384x14x14xf32>, tensor<384x1x1xf32>) outs(%496 : tensor<1x384x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.mulf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x384x14x14xf32>
    %expanded_151 = tensor.expand_shape %159 [[0, 1, 2]] output_shape [384, 1, 1] : tensor<384xf32> into tensor<384x1x1xf32>
    %602 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%601, %expanded_151 : tensor<1x384x14x14xf32>, tensor<384x1x1xf32>) outs(%496 : tensor<1x384x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.mulf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x384x14x14xf32>
    %expanded_152 = tensor.expand_shape %160 [[0, 1, 2]] output_shape [384, 1, 1] : tensor<384xf32> into tensor<384x1x1xf32>
    %603 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%602, %expanded_152 : tensor<1x384x14x14xf32>, tensor<384x1x1xf32>) outs(%496 : tensor<1x384x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.addf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x384x14x14xf32>
    %604 = linalg.generic {indexing_maps = [#map1, #map5, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%603, %277 : tensor<1x384x14x14xf32>, tensor<f32>) outs(%496 : tensor<1x384x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.cmpf ogt, %in, %in_246 : f32
      %1281 = arith.select %1280, %in, %in_246 : f32
      linalg.yield %1281 : f32
    } -> tensor<1x384x14x14xf32>
    %605 = linalg.generic {indexing_maps = [#map5, #map1, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%279, %604 : tensor<f32>, tensor<1x384x14x14xf32>) outs(%496 : tensor<1x384x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.cmpf olt, %in, %in_246 : f32
      %1281 = arith.select %1280, %in, %in_246 : f32
      linalg.yield %1281 : f32
    } -> tensor<1x384x14x14xf32>
    %606 = tensor.empty() : tensor<1x96x14x14xf32>
    %607 = linalg.fill ins(%cst_1 : f32) outs(%606 : tensor<1x96x14x14xf32>) -> tensor<1x96x14x14xf32>
    %608 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%605, %161 : tensor<1x384x14x14xf32>, tensor<96x384x1x1xf32>) outs(%607 : tensor<1x96x14x14xf32>) -> tensor<1x96x14x14xf32>
    %609 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%163 : tensor<96xf32>) outs(%305 : tensor<96xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = arith.truncf %cst_3 : f64 to f32
      %1281 = arith.addf %in, %1280 : f32
      linalg.yield %1281 : f32
    } -> tensor<96xf32>
    %610 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%609 : tensor<96xf32>) outs(%305 : tensor<96xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = math.sqrt %in : f32
      linalg.yield %1280 : f32
    } -> tensor<96xf32>
    %611 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%610 : tensor<96xf32>) outs(%305 : tensor<96xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = arith.cmpf one, %in, %cst_1 : f32
      cf.assert %1280, "unimplemented: tensor with zero element"
      %1281 = arith.divf %cst_2, %in : f32
      linalg.yield %1281 : f32
    } -> tensor<96xf32>
    %expanded_153 = tensor.expand_shape %162 [[0, 1, 2]] output_shape [96, 1, 1] : tensor<96xf32> into tensor<96x1x1xf32>
    %expanded_154 = tensor.expand_shape %611 [[0, 1, 2]] output_shape [96, 1, 1] : tensor<96xf32> into tensor<96x1x1xf32>
    %612 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%608, %expanded_153 : tensor<1x96x14x14xf32>, tensor<96x1x1xf32>) outs(%606 : tensor<1x96x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.subf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x96x14x14xf32>
    %613 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%612, %expanded_154 : tensor<1x96x14x14xf32>, tensor<96x1x1xf32>) outs(%606 : tensor<1x96x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.mulf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x96x14x14xf32>
    %expanded_155 = tensor.expand_shape %164 [[0, 1, 2]] output_shape [96, 1, 1] : tensor<96xf32> into tensor<96x1x1xf32>
    %614 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%613, %expanded_155 : tensor<1x96x14x14xf32>, tensor<96x1x1xf32>) outs(%606 : tensor<1x96x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.mulf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x96x14x14xf32>
    %expanded_156 = tensor.expand_shape %165 [[0, 1, 2]] output_shape [96, 1, 1] : tensor<96xf32> into tensor<96x1x1xf32>
    %615 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%614, %expanded_156 : tensor<1x96x14x14xf32>, tensor<96x1x1xf32>) outs(%606 : tensor<1x96x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.addf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x96x14x14xf32>
    %616 = tensor.empty() : tensor<1x576x14x14xf32>
    %617 = linalg.fill ins(%cst_1 : f32) outs(%616 : tensor<1x576x14x14xf32>) -> tensor<1x576x14x14xf32>
    %618 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%615, %166 : tensor<1x96x14x14xf32>, tensor<576x96x1x1xf32>) outs(%617 : tensor<1x576x14x14xf32>) -> tensor<1x576x14x14xf32>
    %619 = tensor.empty() : tensor<576xf32>
    %620 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%168 : tensor<576xf32>) outs(%619 : tensor<576xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = arith.truncf %cst_3 : f64 to f32
      %1281 = arith.addf %in, %1280 : f32
      linalg.yield %1281 : f32
    } -> tensor<576xf32>
    %621 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%620 : tensor<576xf32>) outs(%619 : tensor<576xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = math.sqrt %in : f32
      linalg.yield %1280 : f32
    } -> tensor<576xf32>
    %622 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%621 : tensor<576xf32>) outs(%619 : tensor<576xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = arith.cmpf one, %in, %cst_1 : f32
      cf.assert %1280, "unimplemented: tensor with zero element"
      %1281 = arith.divf %cst_2, %in : f32
      linalg.yield %1281 : f32
    } -> tensor<576xf32>
    %expanded_157 = tensor.expand_shape %167 [[0, 1, 2]] output_shape [576, 1, 1] : tensor<576xf32> into tensor<576x1x1xf32>
    %expanded_158 = tensor.expand_shape %622 [[0, 1, 2]] output_shape [576, 1, 1] : tensor<576xf32> into tensor<576x1x1xf32>
    %623 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%618, %expanded_157 : tensor<1x576x14x14xf32>, tensor<576x1x1xf32>) outs(%616 : tensor<1x576x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.subf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x576x14x14xf32>
    %624 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%623, %expanded_158 : tensor<1x576x14x14xf32>, tensor<576x1x1xf32>) outs(%616 : tensor<1x576x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.mulf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x576x14x14xf32>
    %expanded_159 = tensor.expand_shape %169 [[0, 1, 2]] output_shape [576, 1, 1] : tensor<576xf32> into tensor<576x1x1xf32>
    %625 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%624, %expanded_159 : tensor<1x576x14x14xf32>, tensor<576x1x1xf32>) outs(%616 : tensor<1x576x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.mulf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x576x14x14xf32>
    %expanded_160 = tensor.expand_shape %170 [[0, 1, 2]] output_shape [576, 1, 1] : tensor<576xf32> into tensor<576x1x1xf32>
    %626 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%625, %expanded_160 : tensor<1x576x14x14xf32>, tensor<576x1x1xf32>) outs(%616 : tensor<1x576x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.addf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x576x14x14xf32>
    %627 = linalg.generic {indexing_maps = [#map1, #map5, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%626, %277 : tensor<1x576x14x14xf32>, tensor<f32>) outs(%616 : tensor<1x576x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.cmpf ogt, %in, %in_246 : f32
      %1281 = arith.select %1280, %in, %in_246 : f32
      linalg.yield %1281 : f32
    } -> tensor<1x576x14x14xf32>
    %628 = linalg.generic {indexing_maps = [#map5, #map1, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%279, %627 : tensor<f32>, tensor<1x576x14x14xf32>) outs(%616 : tensor<1x576x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.cmpf olt, %in, %in_246 : f32
      %1281 = arith.select %1280, %in, %in_246 : f32
      linalg.yield %1281 : f32
    } -> tensor<1x576x14x14xf32>
    %padded_161 = tensor.pad %628 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg265: index, %arg266: index, %arg267: index, %arg268: index):
      tensor.yield %cst_1 : f32
    } : tensor<1x576x14x14xf32> to tensor<1x576x16x16xf32>
    %collapsed_162 = tensor.collapse_shape %171 [[0, 1], [2], [3]] : tensor<576x1x3x3xf32> into tensor<576x3x3xf32>
    %629 = linalg.depthwise_conv_2d_nchw_chw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_161, %collapsed_162 : tensor<1x576x16x16xf32>, tensor<576x3x3xf32>) outs(%617 : tensor<1x576x14x14xf32>) -> tensor<1x576x14x14xf32>
    %630 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%173 : tensor<576xf32>) outs(%619 : tensor<576xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = arith.truncf %cst_3 : f64 to f32
      %1281 = arith.addf %in, %1280 : f32
      linalg.yield %1281 : f32
    } -> tensor<576xf32>
    %631 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%630 : tensor<576xf32>) outs(%619 : tensor<576xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = math.sqrt %in : f32
      linalg.yield %1280 : f32
    } -> tensor<576xf32>
    %632 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%631 : tensor<576xf32>) outs(%619 : tensor<576xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = arith.cmpf one, %in, %cst_1 : f32
      cf.assert %1280, "unimplemented: tensor with zero element"
      %1281 = arith.divf %cst_2, %in : f32
      linalg.yield %1281 : f32
    } -> tensor<576xf32>
    %expanded_163 = tensor.expand_shape %172 [[0, 1, 2]] output_shape [576, 1, 1] : tensor<576xf32> into tensor<576x1x1xf32>
    %expanded_164 = tensor.expand_shape %632 [[0, 1, 2]] output_shape [576, 1, 1] : tensor<576xf32> into tensor<576x1x1xf32>
    %633 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%629, %expanded_163 : tensor<1x576x14x14xf32>, tensor<576x1x1xf32>) outs(%616 : tensor<1x576x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.subf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x576x14x14xf32>
    %634 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%633, %expanded_164 : tensor<1x576x14x14xf32>, tensor<576x1x1xf32>) outs(%616 : tensor<1x576x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.mulf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x576x14x14xf32>
    %expanded_165 = tensor.expand_shape %174 [[0, 1, 2]] output_shape [576, 1, 1] : tensor<576xf32> into tensor<576x1x1xf32>
    %635 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%634, %expanded_165 : tensor<1x576x14x14xf32>, tensor<576x1x1xf32>) outs(%616 : tensor<1x576x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.mulf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x576x14x14xf32>
    %expanded_166 = tensor.expand_shape %175 [[0, 1, 2]] output_shape [576, 1, 1] : tensor<576xf32> into tensor<576x1x1xf32>
    %636 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%635, %expanded_166 : tensor<1x576x14x14xf32>, tensor<576x1x1xf32>) outs(%616 : tensor<1x576x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.addf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x576x14x14xf32>
    %637 = linalg.generic {indexing_maps = [#map1, #map5, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%636, %277 : tensor<1x576x14x14xf32>, tensor<f32>) outs(%616 : tensor<1x576x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.cmpf ogt, %in, %in_246 : f32
      %1281 = arith.select %1280, %in, %in_246 : f32
      linalg.yield %1281 : f32
    } -> tensor<1x576x14x14xf32>
    %638 = linalg.generic {indexing_maps = [#map5, #map1, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%279, %637 : tensor<f32>, tensor<1x576x14x14xf32>) outs(%616 : tensor<1x576x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.cmpf olt, %in, %in_246 : f32
      %1281 = arith.select %1280, %in, %in_246 : f32
      linalg.yield %1281 : f32
    } -> tensor<1x576x14x14xf32>
    %639 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%638, %176 : tensor<1x576x14x14xf32>, tensor<96x576x1x1xf32>) outs(%607 : tensor<1x96x14x14xf32>) -> tensor<1x96x14x14xf32>
    %640 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%178 : tensor<96xf32>) outs(%305 : tensor<96xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = arith.truncf %cst_3 : f64 to f32
      %1281 = arith.addf %in, %1280 : f32
      linalg.yield %1281 : f32
    } -> tensor<96xf32>
    %641 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%640 : tensor<96xf32>) outs(%305 : tensor<96xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = math.sqrt %in : f32
      linalg.yield %1280 : f32
    } -> tensor<96xf32>
    %642 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%641 : tensor<96xf32>) outs(%305 : tensor<96xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = arith.cmpf one, %in, %cst_1 : f32
      cf.assert %1280, "unimplemented: tensor with zero element"
      %1281 = arith.divf %cst_2, %in : f32
      linalg.yield %1281 : f32
    } -> tensor<96xf32>
    %expanded_167 = tensor.expand_shape %177 [[0, 1, 2]] output_shape [96, 1, 1] : tensor<96xf32> into tensor<96x1x1xf32>
    %expanded_168 = tensor.expand_shape %642 [[0, 1, 2]] output_shape [96, 1, 1] : tensor<96xf32> into tensor<96x1x1xf32>
    %643 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%639, %expanded_167 : tensor<1x96x14x14xf32>, tensor<96x1x1xf32>) outs(%606 : tensor<1x96x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.subf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x96x14x14xf32>
    %644 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%643, %expanded_168 : tensor<1x96x14x14xf32>, tensor<96x1x1xf32>) outs(%606 : tensor<1x96x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.mulf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x96x14x14xf32>
    %expanded_169 = tensor.expand_shape %179 [[0, 1, 2]] output_shape [96, 1, 1] : tensor<96xf32> into tensor<96x1x1xf32>
    %645 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%644, %expanded_169 : tensor<1x96x14x14xf32>, tensor<96x1x1xf32>) outs(%606 : tensor<1x96x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.mulf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x96x14x14xf32>
    %expanded_170 = tensor.expand_shape %180 [[0, 1, 2]] output_shape [96, 1, 1] : tensor<96xf32> into tensor<96x1x1xf32>
    %646 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%645, %expanded_170 : tensor<1x96x14x14xf32>, tensor<96x1x1xf32>) outs(%606 : tensor<1x96x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.addf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x96x14x14xf32>
    %647 = linalg.generic {indexing_maps = [#map1, #map1, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%615, %646 : tensor<1x96x14x14xf32>, tensor<1x96x14x14xf32>) outs(%606 : tensor<1x96x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.addf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x96x14x14xf32>
    %648 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%647, %181 : tensor<1x96x14x14xf32>, tensor<576x96x1x1xf32>) outs(%617 : tensor<1x576x14x14xf32>) -> tensor<1x576x14x14xf32>
    %649 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%183 : tensor<576xf32>) outs(%619 : tensor<576xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = arith.truncf %cst_3 : f64 to f32
      %1281 = arith.addf %in, %1280 : f32
      linalg.yield %1281 : f32
    } -> tensor<576xf32>
    %650 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%649 : tensor<576xf32>) outs(%619 : tensor<576xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = math.sqrt %in : f32
      linalg.yield %1280 : f32
    } -> tensor<576xf32>
    %651 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%650 : tensor<576xf32>) outs(%619 : tensor<576xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = arith.cmpf one, %in, %cst_1 : f32
      cf.assert %1280, "unimplemented: tensor with zero element"
      %1281 = arith.divf %cst_2, %in : f32
      linalg.yield %1281 : f32
    } -> tensor<576xf32>
    %expanded_171 = tensor.expand_shape %182 [[0, 1, 2]] output_shape [576, 1, 1] : tensor<576xf32> into tensor<576x1x1xf32>
    %expanded_172 = tensor.expand_shape %651 [[0, 1, 2]] output_shape [576, 1, 1] : tensor<576xf32> into tensor<576x1x1xf32>
    %652 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%648, %expanded_171 : tensor<1x576x14x14xf32>, tensor<576x1x1xf32>) outs(%616 : tensor<1x576x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.subf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x576x14x14xf32>
    %653 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%652, %expanded_172 : tensor<1x576x14x14xf32>, tensor<576x1x1xf32>) outs(%616 : tensor<1x576x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.mulf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x576x14x14xf32>
    %expanded_173 = tensor.expand_shape %184 [[0, 1, 2]] output_shape [576, 1, 1] : tensor<576xf32> into tensor<576x1x1xf32>
    %654 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%653, %expanded_173 : tensor<1x576x14x14xf32>, tensor<576x1x1xf32>) outs(%616 : tensor<1x576x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.mulf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x576x14x14xf32>
    %expanded_174 = tensor.expand_shape %185 [[0, 1, 2]] output_shape [576, 1, 1] : tensor<576xf32> into tensor<576x1x1xf32>
    %655 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%654, %expanded_174 : tensor<1x576x14x14xf32>, tensor<576x1x1xf32>) outs(%616 : tensor<1x576x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.addf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x576x14x14xf32>
    %656 = linalg.generic {indexing_maps = [#map1, #map5, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%655, %277 : tensor<1x576x14x14xf32>, tensor<f32>) outs(%616 : tensor<1x576x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.cmpf ogt, %in, %in_246 : f32
      %1281 = arith.select %1280, %in, %in_246 : f32
      linalg.yield %1281 : f32
    } -> tensor<1x576x14x14xf32>
    %657 = linalg.generic {indexing_maps = [#map5, #map1, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%279, %656 : tensor<f32>, tensor<1x576x14x14xf32>) outs(%616 : tensor<1x576x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.cmpf olt, %in, %in_246 : f32
      %1281 = arith.select %1280, %in, %in_246 : f32
      linalg.yield %1281 : f32
    } -> tensor<1x576x14x14xf32>
    %padded_175 = tensor.pad %657 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg265: index, %arg266: index, %arg267: index, %arg268: index):
      tensor.yield %cst_1 : f32
    } : tensor<1x576x14x14xf32> to tensor<1x576x16x16xf32>
    %collapsed_176 = tensor.collapse_shape %186 [[0, 1], [2], [3]] : tensor<576x1x3x3xf32> into tensor<576x3x3xf32>
    %658 = linalg.depthwise_conv_2d_nchw_chw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_175, %collapsed_176 : tensor<1x576x16x16xf32>, tensor<576x3x3xf32>) outs(%617 : tensor<1x576x14x14xf32>) -> tensor<1x576x14x14xf32>
    %659 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%188 : tensor<576xf32>) outs(%619 : tensor<576xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = arith.truncf %cst_3 : f64 to f32
      %1281 = arith.addf %in, %1280 : f32
      linalg.yield %1281 : f32
    } -> tensor<576xf32>
    %660 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%659 : tensor<576xf32>) outs(%619 : tensor<576xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = math.sqrt %in : f32
      linalg.yield %1280 : f32
    } -> tensor<576xf32>
    %661 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%660 : tensor<576xf32>) outs(%619 : tensor<576xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = arith.cmpf one, %in, %cst_1 : f32
      cf.assert %1280, "unimplemented: tensor with zero element"
      %1281 = arith.divf %cst_2, %in : f32
      linalg.yield %1281 : f32
    } -> tensor<576xf32>
    %expanded_177 = tensor.expand_shape %187 [[0, 1, 2]] output_shape [576, 1, 1] : tensor<576xf32> into tensor<576x1x1xf32>
    %expanded_178 = tensor.expand_shape %661 [[0, 1, 2]] output_shape [576, 1, 1] : tensor<576xf32> into tensor<576x1x1xf32>
    %662 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%658, %expanded_177 : tensor<1x576x14x14xf32>, tensor<576x1x1xf32>) outs(%616 : tensor<1x576x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.subf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x576x14x14xf32>
    %663 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%662, %expanded_178 : tensor<1x576x14x14xf32>, tensor<576x1x1xf32>) outs(%616 : tensor<1x576x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.mulf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x576x14x14xf32>
    %expanded_179 = tensor.expand_shape %189 [[0, 1, 2]] output_shape [576, 1, 1] : tensor<576xf32> into tensor<576x1x1xf32>
    %664 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%663, %expanded_179 : tensor<1x576x14x14xf32>, tensor<576x1x1xf32>) outs(%616 : tensor<1x576x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.mulf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x576x14x14xf32>
    %expanded_180 = tensor.expand_shape %190 [[0, 1, 2]] output_shape [576, 1, 1] : tensor<576xf32> into tensor<576x1x1xf32>
    %665 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%664, %expanded_180 : tensor<1x576x14x14xf32>, tensor<576x1x1xf32>) outs(%616 : tensor<1x576x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.addf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x576x14x14xf32>
    %666 = linalg.generic {indexing_maps = [#map1, #map5, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%665, %277 : tensor<1x576x14x14xf32>, tensor<f32>) outs(%616 : tensor<1x576x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.cmpf ogt, %in, %in_246 : f32
      %1281 = arith.select %1280, %in, %in_246 : f32
      linalg.yield %1281 : f32
    } -> tensor<1x576x14x14xf32>
    %667 = linalg.generic {indexing_maps = [#map5, #map1, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%279, %666 : tensor<f32>, tensor<1x576x14x14xf32>) outs(%616 : tensor<1x576x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.cmpf olt, %in, %in_246 : f32
      %1281 = arith.select %1280, %in, %in_246 : f32
      linalg.yield %1281 : f32
    } -> tensor<1x576x14x14xf32>
    %668 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%667, %191 : tensor<1x576x14x14xf32>, tensor<96x576x1x1xf32>) outs(%607 : tensor<1x96x14x14xf32>) -> tensor<1x96x14x14xf32>
    %669 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%193 : tensor<96xf32>) outs(%305 : tensor<96xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = arith.truncf %cst_3 : f64 to f32
      %1281 = arith.addf %in, %1280 : f32
      linalg.yield %1281 : f32
    } -> tensor<96xf32>
    %670 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%669 : tensor<96xf32>) outs(%305 : tensor<96xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = math.sqrt %in : f32
      linalg.yield %1280 : f32
    } -> tensor<96xf32>
    %671 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%670 : tensor<96xf32>) outs(%305 : tensor<96xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = arith.cmpf one, %in, %cst_1 : f32
      cf.assert %1280, "unimplemented: tensor with zero element"
      %1281 = arith.divf %cst_2, %in : f32
      linalg.yield %1281 : f32
    } -> tensor<96xf32>
    %expanded_181 = tensor.expand_shape %192 [[0, 1, 2]] output_shape [96, 1, 1] : tensor<96xf32> into tensor<96x1x1xf32>
    %expanded_182 = tensor.expand_shape %671 [[0, 1, 2]] output_shape [96, 1, 1] : tensor<96xf32> into tensor<96x1x1xf32>
    %672 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%668, %expanded_181 : tensor<1x96x14x14xf32>, tensor<96x1x1xf32>) outs(%606 : tensor<1x96x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.subf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x96x14x14xf32>
    %673 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%672, %expanded_182 : tensor<1x96x14x14xf32>, tensor<96x1x1xf32>) outs(%606 : tensor<1x96x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.mulf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x96x14x14xf32>
    %expanded_183 = tensor.expand_shape %194 [[0, 1, 2]] output_shape [96, 1, 1] : tensor<96xf32> into tensor<96x1x1xf32>
    %674 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%673, %expanded_183 : tensor<1x96x14x14xf32>, tensor<96x1x1xf32>) outs(%606 : tensor<1x96x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.mulf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x96x14x14xf32>
    %expanded_184 = tensor.expand_shape %195 [[0, 1, 2]] output_shape [96, 1, 1] : tensor<96xf32> into tensor<96x1x1xf32>
    %675 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%674, %expanded_184 : tensor<1x96x14x14xf32>, tensor<96x1x1xf32>) outs(%606 : tensor<1x96x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.addf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x96x14x14xf32>
    %676 = linalg.generic {indexing_maps = [#map1, #map1, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%647, %675 : tensor<1x96x14x14xf32>, tensor<1x96x14x14xf32>) outs(%606 : tensor<1x96x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.addf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x96x14x14xf32>
    %677 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%676, %196 : tensor<1x96x14x14xf32>, tensor<576x96x1x1xf32>) outs(%617 : tensor<1x576x14x14xf32>) -> tensor<1x576x14x14xf32>
    %678 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%198 : tensor<576xf32>) outs(%619 : tensor<576xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = arith.truncf %cst_3 : f64 to f32
      %1281 = arith.addf %in, %1280 : f32
      linalg.yield %1281 : f32
    } -> tensor<576xf32>
    %679 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%678 : tensor<576xf32>) outs(%619 : tensor<576xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = math.sqrt %in : f32
      linalg.yield %1280 : f32
    } -> tensor<576xf32>
    %680 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%679 : tensor<576xf32>) outs(%619 : tensor<576xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = arith.cmpf one, %in, %cst_1 : f32
      cf.assert %1280, "unimplemented: tensor with zero element"
      %1281 = arith.divf %cst_2, %in : f32
      linalg.yield %1281 : f32
    } -> tensor<576xf32>
    %expanded_185 = tensor.expand_shape %197 [[0, 1, 2]] output_shape [576, 1, 1] : tensor<576xf32> into tensor<576x1x1xf32>
    %expanded_186 = tensor.expand_shape %680 [[0, 1, 2]] output_shape [576, 1, 1] : tensor<576xf32> into tensor<576x1x1xf32>
    %681 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%677, %expanded_185 : tensor<1x576x14x14xf32>, tensor<576x1x1xf32>) outs(%616 : tensor<1x576x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.subf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x576x14x14xf32>
    %682 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%681, %expanded_186 : tensor<1x576x14x14xf32>, tensor<576x1x1xf32>) outs(%616 : tensor<1x576x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.mulf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x576x14x14xf32>
    %expanded_187 = tensor.expand_shape %199 [[0, 1, 2]] output_shape [576, 1, 1] : tensor<576xf32> into tensor<576x1x1xf32>
    %683 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%682, %expanded_187 : tensor<1x576x14x14xf32>, tensor<576x1x1xf32>) outs(%616 : tensor<1x576x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.mulf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x576x14x14xf32>
    %expanded_188 = tensor.expand_shape %200 [[0, 1, 2]] output_shape [576, 1, 1] : tensor<576xf32> into tensor<576x1x1xf32>
    %684 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%683, %expanded_188 : tensor<1x576x14x14xf32>, tensor<576x1x1xf32>) outs(%616 : tensor<1x576x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.addf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x576x14x14xf32>
    %685 = linalg.generic {indexing_maps = [#map1, #map5, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%684, %277 : tensor<1x576x14x14xf32>, tensor<f32>) outs(%616 : tensor<1x576x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.cmpf ogt, %in, %in_246 : f32
      %1281 = arith.select %1280, %in, %in_246 : f32
      linalg.yield %1281 : f32
    } -> tensor<1x576x14x14xf32>
    %686 = linalg.generic {indexing_maps = [#map5, #map1, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%279, %685 : tensor<f32>, tensor<1x576x14x14xf32>) outs(%616 : tensor<1x576x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.cmpf olt, %in, %in_246 : f32
      %1281 = arith.select %1280, %in, %in_246 : f32
      linalg.yield %1281 : f32
    } -> tensor<1x576x14x14xf32>
    %padded_189 = tensor.pad %686 low[0, 0, 0, 0] high[0, 0, 1, 1] {
    ^bb0(%arg265: index, %arg266: index, %arg267: index, %arg268: index):
      tensor.yield %cst_1 : f32
    } : tensor<1x576x14x14xf32> to tensor<1x576x15x15xf32>
    %687 = tensor.empty() : tensor<1x576x7x7xf32>
    %688 = linalg.fill ins(%cst_1 : f32) outs(%687 : tensor<1x576x7x7xf32>) -> tensor<1x576x7x7xf32>
    %collapsed_190 = tensor.collapse_shape %201 [[0, 1], [2], [3]] : tensor<576x1x3x3xf32> into tensor<576x3x3xf32>
    %689 = linalg.depthwise_conv_2d_nchw_chw {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%padded_189, %collapsed_190 : tensor<1x576x15x15xf32>, tensor<576x3x3xf32>) outs(%688 : tensor<1x576x7x7xf32>) -> tensor<1x576x7x7xf32>
    %690 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%203 : tensor<576xf32>) outs(%619 : tensor<576xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = arith.truncf %cst_3 : f64 to f32
      %1281 = arith.addf %in, %1280 : f32
      linalg.yield %1281 : f32
    } -> tensor<576xf32>
    %691 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%690 : tensor<576xf32>) outs(%619 : tensor<576xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = math.sqrt %in : f32
      linalg.yield %1280 : f32
    } -> tensor<576xf32>
    %692 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%691 : tensor<576xf32>) outs(%619 : tensor<576xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = arith.cmpf one, %in, %cst_1 : f32
      cf.assert %1280, "unimplemented: tensor with zero element"
      %1281 = arith.divf %cst_2, %in : f32
      linalg.yield %1281 : f32
    } -> tensor<576xf32>
    %expanded_191 = tensor.expand_shape %202 [[0, 1, 2]] output_shape [576, 1, 1] : tensor<576xf32> into tensor<576x1x1xf32>
    %expanded_192 = tensor.expand_shape %692 [[0, 1, 2]] output_shape [576, 1, 1] : tensor<576xf32> into tensor<576x1x1xf32>
    %693 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%689, %expanded_191 : tensor<1x576x7x7xf32>, tensor<576x1x1xf32>) outs(%687 : tensor<1x576x7x7xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.subf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x576x7x7xf32>
    %694 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%693, %expanded_192 : tensor<1x576x7x7xf32>, tensor<576x1x1xf32>) outs(%687 : tensor<1x576x7x7xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.mulf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x576x7x7xf32>
    %expanded_193 = tensor.expand_shape %204 [[0, 1, 2]] output_shape [576, 1, 1] : tensor<576xf32> into tensor<576x1x1xf32>
    %695 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%694, %expanded_193 : tensor<1x576x7x7xf32>, tensor<576x1x1xf32>) outs(%687 : tensor<1x576x7x7xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.mulf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x576x7x7xf32>
    %expanded_194 = tensor.expand_shape %205 [[0, 1, 2]] output_shape [576, 1, 1] : tensor<576xf32> into tensor<576x1x1xf32>
    %696 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%695, %expanded_194 : tensor<1x576x7x7xf32>, tensor<576x1x1xf32>) outs(%687 : tensor<1x576x7x7xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.addf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x576x7x7xf32>
    %697 = linalg.generic {indexing_maps = [#map1, #map5, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%696, %277 : tensor<1x576x7x7xf32>, tensor<f32>) outs(%687 : tensor<1x576x7x7xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.cmpf ogt, %in, %in_246 : f32
      %1281 = arith.select %1280, %in, %in_246 : f32
      linalg.yield %1281 : f32
    } -> tensor<1x576x7x7xf32>
    %698 = linalg.generic {indexing_maps = [#map5, #map1, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%279, %697 : tensor<f32>, tensor<1x576x7x7xf32>) outs(%687 : tensor<1x576x7x7xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.cmpf olt, %in, %in_246 : f32
      %1281 = arith.select %1280, %in, %in_246 : f32
      linalg.yield %1281 : f32
    } -> tensor<1x576x7x7xf32>
    %699 = tensor.empty() : tensor<1x160x7x7xf32>
    %700 = linalg.fill ins(%cst_1 : f32) outs(%699 : tensor<1x160x7x7xf32>) -> tensor<1x160x7x7xf32>
    %701 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%698, %206 : tensor<1x576x7x7xf32>, tensor<160x576x1x1xf32>) outs(%700 : tensor<1x160x7x7xf32>) -> tensor<1x160x7x7xf32>
    %702 = tensor.empty() : tensor<160xf32>
    %703 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%208 : tensor<160xf32>) outs(%702 : tensor<160xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = arith.truncf %cst_3 : f64 to f32
      %1281 = arith.addf %in, %1280 : f32
      linalg.yield %1281 : f32
    } -> tensor<160xf32>
    %704 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%703 : tensor<160xf32>) outs(%702 : tensor<160xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = math.sqrt %in : f32
      linalg.yield %1280 : f32
    } -> tensor<160xf32>
    %705 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%704 : tensor<160xf32>) outs(%702 : tensor<160xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = arith.cmpf one, %in, %cst_1 : f32
      cf.assert %1280, "unimplemented: tensor with zero element"
      %1281 = arith.divf %cst_2, %in : f32
      linalg.yield %1281 : f32
    } -> tensor<160xf32>
    %expanded_195 = tensor.expand_shape %207 [[0, 1, 2]] output_shape [160, 1, 1] : tensor<160xf32> into tensor<160x1x1xf32>
    %expanded_196 = tensor.expand_shape %705 [[0, 1, 2]] output_shape [160, 1, 1] : tensor<160xf32> into tensor<160x1x1xf32>
    %706 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%701, %expanded_195 : tensor<1x160x7x7xf32>, tensor<160x1x1xf32>) outs(%699 : tensor<1x160x7x7xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.subf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x160x7x7xf32>
    %707 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%706, %expanded_196 : tensor<1x160x7x7xf32>, tensor<160x1x1xf32>) outs(%699 : tensor<1x160x7x7xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.mulf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x160x7x7xf32>
    %expanded_197 = tensor.expand_shape %209 [[0, 1, 2]] output_shape [160, 1, 1] : tensor<160xf32> into tensor<160x1x1xf32>
    %708 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%707, %expanded_197 : tensor<1x160x7x7xf32>, tensor<160x1x1xf32>) outs(%699 : tensor<1x160x7x7xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.mulf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x160x7x7xf32>
    %expanded_198 = tensor.expand_shape %210 [[0, 1, 2]] output_shape [160, 1, 1] : tensor<160xf32> into tensor<160x1x1xf32>
    %709 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%708, %expanded_198 : tensor<1x160x7x7xf32>, tensor<160x1x1xf32>) outs(%699 : tensor<1x160x7x7xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.addf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x160x7x7xf32>
    %710 = tensor.empty() : tensor<1x960x7x7xf32>
    %711 = linalg.fill ins(%cst_1 : f32) outs(%710 : tensor<1x960x7x7xf32>) -> tensor<1x960x7x7xf32>
    %712 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%709, %211 : tensor<1x160x7x7xf32>, tensor<960x160x1x1xf32>) outs(%711 : tensor<1x960x7x7xf32>) -> tensor<1x960x7x7xf32>
    %713 = tensor.empty() : tensor<960xf32>
    %714 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%213 : tensor<960xf32>) outs(%713 : tensor<960xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = arith.truncf %cst_3 : f64 to f32
      %1281 = arith.addf %in, %1280 : f32
      linalg.yield %1281 : f32
    } -> tensor<960xf32>
    %715 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%714 : tensor<960xf32>) outs(%713 : tensor<960xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = math.sqrt %in : f32
      linalg.yield %1280 : f32
    } -> tensor<960xf32>
    %716 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%715 : tensor<960xf32>) outs(%713 : tensor<960xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = arith.cmpf one, %in, %cst_1 : f32
      cf.assert %1280, "unimplemented: tensor with zero element"
      %1281 = arith.divf %cst_2, %in : f32
      linalg.yield %1281 : f32
    } -> tensor<960xf32>
    %expanded_199 = tensor.expand_shape %212 [[0, 1, 2]] output_shape [960, 1, 1] : tensor<960xf32> into tensor<960x1x1xf32>
    %expanded_200 = tensor.expand_shape %716 [[0, 1, 2]] output_shape [960, 1, 1] : tensor<960xf32> into tensor<960x1x1xf32>
    %717 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%712, %expanded_199 : tensor<1x960x7x7xf32>, tensor<960x1x1xf32>) outs(%710 : tensor<1x960x7x7xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.subf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x960x7x7xf32>
    %718 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%717, %expanded_200 : tensor<1x960x7x7xf32>, tensor<960x1x1xf32>) outs(%710 : tensor<1x960x7x7xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.mulf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x960x7x7xf32>
    %expanded_201 = tensor.expand_shape %214 [[0, 1, 2]] output_shape [960, 1, 1] : tensor<960xf32> into tensor<960x1x1xf32>
    %719 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%718, %expanded_201 : tensor<1x960x7x7xf32>, tensor<960x1x1xf32>) outs(%710 : tensor<1x960x7x7xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.mulf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x960x7x7xf32>
    %expanded_202 = tensor.expand_shape %215 [[0, 1, 2]] output_shape [960, 1, 1] : tensor<960xf32> into tensor<960x1x1xf32>
    %720 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%719, %expanded_202 : tensor<1x960x7x7xf32>, tensor<960x1x1xf32>) outs(%710 : tensor<1x960x7x7xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.addf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x960x7x7xf32>
    %721 = linalg.generic {indexing_maps = [#map1, #map5, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%720, %277 : tensor<1x960x7x7xf32>, tensor<f32>) outs(%710 : tensor<1x960x7x7xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.cmpf ogt, %in, %in_246 : f32
      %1281 = arith.select %1280, %in, %in_246 : f32
      linalg.yield %1281 : f32
    } -> tensor<1x960x7x7xf32>
    %722 = linalg.generic {indexing_maps = [#map5, #map1, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%279, %721 : tensor<f32>, tensor<1x960x7x7xf32>) outs(%710 : tensor<1x960x7x7xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.cmpf olt, %in, %in_246 : f32
      %1281 = arith.select %1280, %in, %in_246 : f32
      linalg.yield %1281 : f32
    } -> tensor<1x960x7x7xf32>
    %padded_203 = tensor.pad %722 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg265: index, %arg266: index, %arg267: index, %arg268: index):
      tensor.yield %cst_1 : f32
    } : tensor<1x960x7x7xf32> to tensor<1x960x9x9xf32>
    %collapsed_204 = tensor.collapse_shape %216 [[0, 1], [2], [3]] : tensor<960x1x3x3xf32> into tensor<960x3x3xf32>
    %723 = linalg.depthwise_conv_2d_nchw_chw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_203, %collapsed_204 : tensor<1x960x9x9xf32>, tensor<960x3x3xf32>) outs(%711 : tensor<1x960x7x7xf32>) -> tensor<1x960x7x7xf32>
    %724 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%218 : tensor<960xf32>) outs(%713 : tensor<960xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = arith.truncf %cst_3 : f64 to f32
      %1281 = arith.addf %in, %1280 : f32
      linalg.yield %1281 : f32
    } -> tensor<960xf32>
    %725 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%724 : tensor<960xf32>) outs(%713 : tensor<960xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = math.sqrt %in : f32
      linalg.yield %1280 : f32
    } -> tensor<960xf32>
    %726 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%725 : tensor<960xf32>) outs(%713 : tensor<960xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = arith.cmpf one, %in, %cst_1 : f32
      cf.assert %1280, "unimplemented: tensor with zero element"
      %1281 = arith.divf %cst_2, %in : f32
      linalg.yield %1281 : f32
    } -> tensor<960xf32>
    %expanded_205 = tensor.expand_shape %217 [[0, 1, 2]] output_shape [960, 1, 1] : tensor<960xf32> into tensor<960x1x1xf32>
    %expanded_206 = tensor.expand_shape %726 [[0, 1, 2]] output_shape [960, 1, 1] : tensor<960xf32> into tensor<960x1x1xf32>
    %727 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%723, %expanded_205 : tensor<1x960x7x7xf32>, tensor<960x1x1xf32>) outs(%710 : tensor<1x960x7x7xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.subf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x960x7x7xf32>
    %728 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%727, %expanded_206 : tensor<1x960x7x7xf32>, tensor<960x1x1xf32>) outs(%710 : tensor<1x960x7x7xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.mulf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x960x7x7xf32>
    %expanded_207 = tensor.expand_shape %219 [[0, 1, 2]] output_shape [960, 1, 1] : tensor<960xf32> into tensor<960x1x1xf32>
    %729 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%728, %expanded_207 : tensor<1x960x7x7xf32>, tensor<960x1x1xf32>) outs(%710 : tensor<1x960x7x7xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.mulf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x960x7x7xf32>
    %expanded_208 = tensor.expand_shape %220 [[0, 1, 2]] output_shape [960, 1, 1] : tensor<960xf32> into tensor<960x1x1xf32>
    %730 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%729, %expanded_208 : tensor<1x960x7x7xf32>, tensor<960x1x1xf32>) outs(%710 : tensor<1x960x7x7xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.addf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x960x7x7xf32>
    %731 = linalg.generic {indexing_maps = [#map1, #map5, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%730, %277 : tensor<1x960x7x7xf32>, tensor<f32>) outs(%710 : tensor<1x960x7x7xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.cmpf ogt, %in, %in_246 : f32
      %1281 = arith.select %1280, %in, %in_246 : f32
      linalg.yield %1281 : f32
    } -> tensor<1x960x7x7xf32>
    %732 = linalg.generic {indexing_maps = [#map5, #map1, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%279, %731 : tensor<f32>, tensor<1x960x7x7xf32>) outs(%710 : tensor<1x960x7x7xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.cmpf olt, %in, %in_246 : f32
      %1281 = arith.select %1280, %in, %in_246 : f32
      linalg.yield %1281 : f32
    } -> tensor<1x960x7x7xf32>
    %733 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%732, %221 : tensor<1x960x7x7xf32>, tensor<160x960x1x1xf32>) outs(%700 : tensor<1x160x7x7xf32>) -> tensor<1x160x7x7xf32>
    %734 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%223 : tensor<160xf32>) outs(%702 : tensor<160xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = arith.truncf %cst_3 : f64 to f32
      %1281 = arith.addf %in, %1280 : f32
      linalg.yield %1281 : f32
    } -> tensor<160xf32>
    %735 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%734 : tensor<160xf32>) outs(%702 : tensor<160xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = math.sqrt %in : f32
      linalg.yield %1280 : f32
    } -> tensor<160xf32>
    %736 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%735 : tensor<160xf32>) outs(%702 : tensor<160xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = arith.cmpf one, %in, %cst_1 : f32
      cf.assert %1280, "unimplemented: tensor with zero element"
      %1281 = arith.divf %cst_2, %in : f32
      linalg.yield %1281 : f32
    } -> tensor<160xf32>
    %expanded_209 = tensor.expand_shape %222 [[0, 1, 2]] output_shape [160, 1, 1] : tensor<160xf32> into tensor<160x1x1xf32>
    %expanded_210 = tensor.expand_shape %736 [[0, 1, 2]] output_shape [160, 1, 1] : tensor<160xf32> into tensor<160x1x1xf32>
    %737 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%733, %expanded_209 : tensor<1x160x7x7xf32>, tensor<160x1x1xf32>) outs(%699 : tensor<1x160x7x7xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.subf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x160x7x7xf32>
    %738 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%737, %expanded_210 : tensor<1x160x7x7xf32>, tensor<160x1x1xf32>) outs(%699 : tensor<1x160x7x7xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.mulf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x160x7x7xf32>
    %expanded_211 = tensor.expand_shape %224 [[0, 1, 2]] output_shape [160, 1, 1] : tensor<160xf32> into tensor<160x1x1xf32>
    %739 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%738, %expanded_211 : tensor<1x160x7x7xf32>, tensor<160x1x1xf32>) outs(%699 : tensor<1x160x7x7xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.mulf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x160x7x7xf32>
    %expanded_212 = tensor.expand_shape %225 [[0, 1, 2]] output_shape [160, 1, 1] : tensor<160xf32> into tensor<160x1x1xf32>
    %740 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%739, %expanded_212 : tensor<1x160x7x7xf32>, tensor<160x1x1xf32>) outs(%699 : tensor<1x160x7x7xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.addf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x160x7x7xf32>
    %741 = linalg.generic {indexing_maps = [#map1, #map1, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%709, %740 : tensor<1x160x7x7xf32>, tensor<1x160x7x7xf32>) outs(%699 : tensor<1x160x7x7xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.addf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x160x7x7xf32>
    %742 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%741, %226 : tensor<1x160x7x7xf32>, tensor<960x160x1x1xf32>) outs(%711 : tensor<1x960x7x7xf32>) -> tensor<1x960x7x7xf32>
    %743 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%228 : tensor<960xf32>) outs(%713 : tensor<960xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = arith.truncf %cst_3 : f64 to f32
      %1281 = arith.addf %in, %1280 : f32
      linalg.yield %1281 : f32
    } -> tensor<960xf32>
    %744 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%743 : tensor<960xf32>) outs(%713 : tensor<960xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = math.sqrt %in : f32
      linalg.yield %1280 : f32
    } -> tensor<960xf32>
    %745 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%744 : tensor<960xf32>) outs(%713 : tensor<960xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = arith.cmpf one, %in, %cst_1 : f32
      cf.assert %1280, "unimplemented: tensor with zero element"
      %1281 = arith.divf %cst_2, %in : f32
      linalg.yield %1281 : f32
    } -> tensor<960xf32>
    %expanded_213 = tensor.expand_shape %227 [[0, 1, 2]] output_shape [960, 1, 1] : tensor<960xf32> into tensor<960x1x1xf32>
    %expanded_214 = tensor.expand_shape %745 [[0, 1, 2]] output_shape [960, 1, 1] : tensor<960xf32> into tensor<960x1x1xf32>
    %746 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%742, %expanded_213 : tensor<1x960x7x7xf32>, tensor<960x1x1xf32>) outs(%710 : tensor<1x960x7x7xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.subf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x960x7x7xf32>
    %747 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%746, %expanded_214 : tensor<1x960x7x7xf32>, tensor<960x1x1xf32>) outs(%710 : tensor<1x960x7x7xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.mulf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x960x7x7xf32>
    %expanded_215 = tensor.expand_shape %229 [[0, 1, 2]] output_shape [960, 1, 1] : tensor<960xf32> into tensor<960x1x1xf32>
    %748 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%747, %expanded_215 : tensor<1x960x7x7xf32>, tensor<960x1x1xf32>) outs(%710 : tensor<1x960x7x7xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.mulf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x960x7x7xf32>
    %expanded_216 = tensor.expand_shape %230 [[0, 1, 2]] output_shape [960, 1, 1] : tensor<960xf32> into tensor<960x1x1xf32>
    %749 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%748, %expanded_216 : tensor<1x960x7x7xf32>, tensor<960x1x1xf32>) outs(%710 : tensor<1x960x7x7xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.addf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x960x7x7xf32>
    %750 = linalg.generic {indexing_maps = [#map1, #map5, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%749, %277 : tensor<1x960x7x7xf32>, tensor<f32>) outs(%710 : tensor<1x960x7x7xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.cmpf ogt, %in, %in_246 : f32
      %1281 = arith.select %1280, %in, %in_246 : f32
      linalg.yield %1281 : f32
    } -> tensor<1x960x7x7xf32>
    %751 = linalg.generic {indexing_maps = [#map5, #map1, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%279, %750 : tensor<f32>, tensor<1x960x7x7xf32>) outs(%710 : tensor<1x960x7x7xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.cmpf olt, %in, %in_246 : f32
      %1281 = arith.select %1280, %in, %in_246 : f32
      linalg.yield %1281 : f32
    } -> tensor<1x960x7x7xf32>
    %padded_217 = tensor.pad %751 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg265: index, %arg266: index, %arg267: index, %arg268: index):
      tensor.yield %cst_1 : f32
    } : tensor<1x960x7x7xf32> to tensor<1x960x9x9xf32>
    %collapsed_218 = tensor.collapse_shape %231 [[0, 1], [2], [3]] : tensor<960x1x3x3xf32> into tensor<960x3x3xf32>
    %752 = linalg.depthwise_conv_2d_nchw_chw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_217, %collapsed_218 : tensor<1x960x9x9xf32>, tensor<960x3x3xf32>) outs(%711 : tensor<1x960x7x7xf32>) -> tensor<1x960x7x7xf32>
    %753 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%233 : tensor<960xf32>) outs(%713 : tensor<960xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = arith.truncf %cst_3 : f64 to f32
      %1281 = arith.addf %in, %1280 : f32
      linalg.yield %1281 : f32
    } -> tensor<960xf32>
    %754 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%753 : tensor<960xf32>) outs(%713 : tensor<960xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = math.sqrt %in : f32
      linalg.yield %1280 : f32
    } -> tensor<960xf32>
    %755 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%754 : tensor<960xf32>) outs(%713 : tensor<960xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = arith.cmpf one, %in, %cst_1 : f32
      cf.assert %1280, "unimplemented: tensor with zero element"
      %1281 = arith.divf %cst_2, %in : f32
      linalg.yield %1281 : f32
    } -> tensor<960xf32>
    %expanded_219 = tensor.expand_shape %232 [[0, 1, 2]] output_shape [960, 1, 1] : tensor<960xf32> into tensor<960x1x1xf32>
    %expanded_220 = tensor.expand_shape %755 [[0, 1, 2]] output_shape [960, 1, 1] : tensor<960xf32> into tensor<960x1x1xf32>
    %756 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%752, %expanded_219 : tensor<1x960x7x7xf32>, tensor<960x1x1xf32>) outs(%710 : tensor<1x960x7x7xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.subf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x960x7x7xf32>
    %757 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%756, %expanded_220 : tensor<1x960x7x7xf32>, tensor<960x1x1xf32>) outs(%710 : tensor<1x960x7x7xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.mulf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x960x7x7xf32>
    %expanded_221 = tensor.expand_shape %234 [[0, 1, 2]] output_shape [960, 1, 1] : tensor<960xf32> into tensor<960x1x1xf32>
    %758 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%757, %expanded_221 : tensor<1x960x7x7xf32>, tensor<960x1x1xf32>) outs(%710 : tensor<1x960x7x7xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.mulf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x960x7x7xf32>
    %expanded_222 = tensor.expand_shape %235 [[0, 1, 2]] output_shape [960, 1, 1] : tensor<960xf32> into tensor<960x1x1xf32>
    %759 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%758, %expanded_222 : tensor<1x960x7x7xf32>, tensor<960x1x1xf32>) outs(%710 : tensor<1x960x7x7xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.addf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x960x7x7xf32>
    %760 = linalg.generic {indexing_maps = [#map1, #map5, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%759, %277 : tensor<1x960x7x7xf32>, tensor<f32>) outs(%710 : tensor<1x960x7x7xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.cmpf ogt, %in, %in_246 : f32
      %1281 = arith.select %1280, %in, %in_246 : f32
      linalg.yield %1281 : f32
    } -> tensor<1x960x7x7xf32>
    %761 = linalg.generic {indexing_maps = [#map5, #map1, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%279, %760 : tensor<f32>, tensor<1x960x7x7xf32>) outs(%710 : tensor<1x960x7x7xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.cmpf olt, %in, %in_246 : f32
      %1281 = arith.select %1280, %in, %in_246 : f32
      linalg.yield %1281 : f32
    } -> tensor<1x960x7x7xf32>
    %762 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%761, %236 : tensor<1x960x7x7xf32>, tensor<160x960x1x1xf32>) outs(%700 : tensor<1x160x7x7xf32>) -> tensor<1x160x7x7xf32>
    %763 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%238 : tensor<160xf32>) outs(%702 : tensor<160xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = arith.truncf %cst_3 : f64 to f32
      %1281 = arith.addf %in, %1280 : f32
      linalg.yield %1281 : f32
    } -> tensor<160xf32>
    %764 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%763 : tensor<160xf32>) outs(%702 : tensor<160xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = math.sqrt %in : f32
      linalg.yield %1280 : f32
    } -> tensor<160xf32>
    %765 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%764 : tensor<160xf32>) outs(%702 : tensor<160xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = arith.cmpf one, %in, %cst_1 : f32
      cf.assert %1280, "unimplemented: tensor with zero element"
      %1281 = arith.divf %cst_2, %in : f32
      linalg.yield %1281 : f32
    } -> tensor<160xf32>
    %expanded_223 = tensor.expand_shape %237 [[0, 1, 2]] output_shape [160, 1, 1] : tensor<160xf32> into tensor<160x1x1xf32>
    %expanded_224 = tensor.expand_shape %765 [[0, 1, 2]] output_shape [160, 1, 1] : tensor<160xf32> into tensor<160x1x1xf32>
    %766 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%762, %expanded_223 : tensor<1x160x7x7xf32>, tensor<160x1x1xf32>) outs(%699 : tensor<1x160x7x7xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.subf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x160x7x7xf32>
    %767 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%766, %expanded_224 : tensor<1x160x7x7xf32>, tensor<160x1x1xf32>) outs(%699 : tensor<1x160x7x7xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.mulf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x160x7x7xf32>
    %expanded_225 = tensor.expand_shape %239 [[0, 1, 2]] output_shape [160, 1, 1] : tensor<160xf32> into tensor<160x1x1xf32>
    %768 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%767, %expanded_225 : tensor<1x160x7x7xf32>, tensor<160x1x1xf32>) outs(%699 : tensor<1x160x7x7xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.mulf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x160x7x7xf32>
    %expanded_226 = tensor.expand_shape %240 [[0, 1, 2]] output_shape [160, 1, 1] : tensor<160xf32> into tensor<160x1x1xf32>
    %769 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%768, %expanded_226 : tensor<1x160x7x7xf32>, tensor<160x1x1xf32>) outs(%699 : tensor<1x160x7x7xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.addf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x160x7x7xf32>
    %770 = linalg.generic {indexing_maps = [#map1, #map1, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%741, %769 : tensor<1x160x7x7xf32>, tensor<1x160x7x7xf32>) outs(%699 : tensor<1x160x7x7xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.addf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x160x7x7xf32>
    %771 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%770, %241 : tensor<1x160x7x7xf32>, tensor<960x160x1x1xf32>) outs(%711 : tensor<1x960x7x7xf32>) -> tensor<1x960x7x7xf32>
    %772 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%243 : tensor<960xf32>) outs(%713 : tensor<960xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = arith.truncf %cst_3 : f64 to f32
      %1281 = arith.addf %in, %1280 : f32
      linalg.yield %1281 : f32
    } -> tensor<960xf32>
    %773 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%772 : tensor<960xf32>) outs(%713 : tensor<960xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = math.sqrt %in : f32
      linalg.yield %1280 : f32
    } -> tensor<960xf32>
    %774 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%773 : tensor<960xf32>) outs(%713 : tensor<960xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = arith.cmpf one, %in, %cst_1 : f32
      cf.assert %1280, "unimplemented: tensor with zero element"
      %1281 = arith.divf %cst_2, %in : f32
      linalg.yield %1281 : f32
    } -> tensor<960xf32>
    %expanded_227 = tensor.expand_shape %242 [[0, 1, 2]] output_shape [960, 1, 1] : tensor<960xf32> into tensor<960x1x1xf32>
    %expanded_228 = tensor.expand_shape %774 [[0, 1, 2]] output_shape [960, 1, 1] : tensor<960xf32> into tensor<960x1x1xf32>
    %775 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%771, %expanded_227 : tensor<1x960x7x7xf32>, tensor<960x1x1xf32>) outs(%710 : tensor<1x960x7x7xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.subf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x960x7x7xf32>
    %776 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%775, %expanded_228 : tensor<1x960x7x7xf32>, tensor<960x1x1xf32>) outs(%710 : tensor<1x960x7x7xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.mulf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x960x7x7xf32>
    %expanded_229 = tensor.expand_shape %244 [[0, 1, 2]] output_shape [960, 1, 1] : tensor<960xf32> into tensor<960x1x1xf32>
    %777 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%776, %expanded_229 : tensor<1x960x7x7xf32>, tensor<960x1x1xf32>) outs(%710 : tensor<1x960x7x7xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.mulf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x960x7x7xf32>
    %expanded_230 = tensor.expand_shape %245 [[0, 1, 2]] output_shape [960, 1, 1] : tensor<960xf32> into tensor<960x1x1xf32>
    %778 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%777, %expanded_230 : tensor<1x960x7x7xf32>, tensor<960x1x1xf32>) outs(%710 : tensor<1x960x7x7xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.addf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x960x7x7xf32>
    %779 = linalg.generic {indexing_maps = [#map1, #map5, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%778, %277 : tensor<1x960x7x7xf32>, tensor<f32>) outs(%710 : tensor<1x960x7x7xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.cmpf ogt, %in, %in_246 : f32
      %1281 = arith.select %1280, %in, %in_246 : f32
      linalg.yield %1281 : f32
    } -> tensor<1x960x7x7xf32>
    %780 = linalg.generic {indexing_maps = [#map5, #map1, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%279, %779 : tensor<f32>, tensor<1x960x7x7xf32>) outs(%710 : tensor<1x960x7x7xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.cmpf olt, %in, %in_246 : f32
      %1281 = arith.select %1280, %in, %in_246 : f32
      linalg.yield %1281 : f32
    } -> tensor<1x960x7x7xf32>
    %padded_231 = tensor.pad %780 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg265: index, %arg266: index, %arg267: index, %arg268: index):
      tensor.yield %cst_1 : f32
    } : tensor<1x960x7x7xf32> to tensor<1x960x9x9xf32>
    %collapsed_232 = tensor.collapse_shape %246 [[0, 1], [2], [3]] : tensor<960x1x3x3xf32> into tensor<960x3x3xf32>
    %781 = linalg.depthwise_conv_2d_nchw_chw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_231, %collapsed_232 : tensor<1x960x9x9xf32>, tensor<960x3x3xf32>) outs(%711 : tensor<1x960x7x7xf32>) -> tensor<1x960x7x7xf32>
    %782 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%248 : tensor<960xf32>) outs(%713 : tensor<960xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = arith.truncf %cst_3 : f64 to f32
      %1281 = arith.addf %in, %1280 : f32
      linalg.yield %1281 : f32
    } -> tensor<960xf32>
    %783 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%782 : tensor<960xf32>) outs(%713 : tensor<960xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = math.sqrt %in : f32
      linalg.yield %1280 : f32
    } -> tensor<960xf32>
    %784 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%783 : tensor<960xf32>) outs(%713 : tensor<960xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = arith.cmpf one, %in, %cst_1 : f32
      cf.assert %1280, "unimplemented: tensor with zero element"
      %1281 = arith.divf %cst_2, %in : f32
      linalg.yield %1281 : f32
    } -> tensor<960xf32>
    %expanded_233 = tensor.expand_shape %247 [[0, 1, 2]] output_shape [960, 1, 1] : tensor<960xf32> into tensor<960x1x1xf32>
    %expanded_234 = tensor.expand_shape %784 [[0, 1, 2]] output_shape [960, 1, 1] : tensor<960xf32> into tensor<960x1x1xf32>
    %785 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%781, %expanded_233 : tensor<1x960x7x7xf32>, tensor<960x1x1xf32>) outs(%710 : tensor<1x960x7x7xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.subf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x960x7x7xf32>
    %786 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%785, %expanded_234 : tensor<1x960x7x7xf32>, tensor<960x1x1xf32>) outs(%710 : tensor<1x960x7x7xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.mulf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x960x7x7xf32>
    %expanded_235 = tensor.expand_shape %249 [[0, 1, 2]] output_shape [960, 1, 1] : tensor<960xf32> into tensor<960x1x1xf32>
    %787 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%786, %expanded_235 : tensor<1x960x7x7xf32>, tensor<960x1x1xf32>) outs(%710 : tensor<1x960x7x7xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.mulf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x960x7x7xf32>
    %expanded_236 = tensor.expand_shape %250 [[0, 1, 2]] output_shape [960, 1, 1] : tensor<960xf32> into tensor<960x1x1xf32>
    %788 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%787, %expanded_236 : tensor<1x960x7x7xf32>, tensor<960x1x1xf32>) outs(%710 : tensor<1x960x7x7xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.addf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x960x7x7xf32>
    %789 = linalg.generic {indexing_maps = [#map1, #map5, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%788, %277 : tensor<1x960x7x7xf32>, tensor<f32>) outs(%710 : tensor<1x960x7x7xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.cmpf ogt, %in, %in_246 : f32
      %1281 = arith.select %1280, %in, %in_246 : f32
      linalg.yield %1281 : f32
    } -> tensor<1x960x7x7xf32>
    %790 = linalg.generic {indexing_maps = [#map5, #map1, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%279, %789 : tensor<f32>, tensor<1x960x7x7xf32>) outs(%710 : tensor<1x960x7x7xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.cmpf olt, %in, %in_246 : f32
      %1281 = arith.select %1280, %in, %in_246 : f32
      linalg.yield %1281 : f32
    } -> tensor<1x960x7x7xf32>
    %791 = tensor.empty() : tensor<1x320x7x7xf32>
    %792 = linalg.fill ins(%cst_1 : f32) outs(%791 : tensor<1x320x7x7xf32>) -> tensor<1x320x7x7xf32>
    %793 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%790, %251 : tensor<1x960x7x7xf32>, tensor<320x960x1x1xf32>) outs(%792 : tensor<1x320x7x7xf32>) -> tensor<1x320x7x7xf32>
    %794 = tensor.empty() : tensor<320xf32>
    %795 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%253 : tensor<320xf32>) outs(%794 : tensor<320xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = arith.truncf %cst_3 : f64 to f32
      %1281 = arith.addf %in, %1280 : f32
      linalg.yield %1281 : f32
    } -> tensor<320xf32>
    %796 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%795 : tensor<320xf32>) outs(%794 : tensor<320xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = math.sqrt %in : f32
      linalg.yield %1280 : f32
    } -> tensor<320xf32>
    %797 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%796 : tensor<320xf32>) outs(%794 : tensor<320xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = arith.cmpf one, %in, %cst_1 : f32
      cf.assert %1280, "unimplemented: tensor with zero element"
      %1281 = arith.divf %cst_2, %in : f32
      linalg.yield %1281 : f32
    } -> tensor<320xf32>
    %expanded_237 = tensor.expand_shape %252 [[0, 1, 2]] output_shape [320, 1, 1] : tensor<320xf32> into tensor<320x1x1xf32>
    %expanded_238 = tensor.expand_shape %797 [[0, 1, 2]] output_shape [320, 1, 1] : tensor<320xf32> into tensor<320x1x1xf32>
    %798 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%793, %expanded_237 : tensor<1x320x7x7xf32>, tensor<320x1x1xf32>) outs(%791 : tensor<1x320x7x7xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.subf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x320x7x7xf32>
    %799 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%798, %expanded_238 : tensor<1x320x7x7xf32>, tensor<320x1x1xf32>) outs(%791 : tensor<1x320x7x7xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.mulf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x320x7x7xf32>
    %expanded_239 = tensor.expand_shape %254 [[0, 1, 2]] output_shape [320, 1, 1] : tensor<320xf32> into tensor<320x1x1xf32>
    %800 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%799, %expanded_239 : tensor<1x320x7x7xf32>, tensor<320x1x1xf32>) outs(%791 : tensor<1x320x7x7xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.mulf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x320x7x7xf32>
    %expanded_240 = tensor.expand_shape %255 [[0, 1, 2]] output_shape [320, 1, 1] : tensor<320xf32> into tensor<320x1x1xf32>
    %801 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%800, %expanded_240 : tensor<1x320x7x7xf32>, tensor<320x1x1xf32>) outs(%791 : tensor<1x320x7x7xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.addf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x320x7x7xf32>
    %802 = tensor.empty() : tensor<1x1280x7x7xf32>
    %803 = linalg.fill ins(%cst_1 : f32) outs(%802 : tensor<1x1280x7x7xf32>) -> tensor<1x1280x7x7xf32>
    %804 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%801, %256 : tensor<1x320x7x7xf32>, tensor<1280x320x1x1xf32>) outs(%803 : tensor<1x1280x7x7xf32>) -> tensor<1x1280x7x7xf32>
    %805 = tensor.empty() : tensor<1280xf32>
    %806 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%258 : tensor<1280xf32>) outs(%805 : tensor<1280xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = arith.truncf %cst_3 : f64 to f32
      %1281 = arith.addf %in, %1280 : f32
      linalg.yield %1281 : f32
    } -> tensor<1280xf32>
    %807 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%806 : tensor<1280xf32>) outs(%805 : tensor<1280xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = math.sqrt %in : f32
      linalg.yield %1280 : f32
    } -> tensor<1280xf32>
    %808 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%807 : tensor<1280xf32>) outs(%805 : tensor<1280xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = arith.cmpf one, %in, %cst_1 : f32
      cf.assert %1280, "unimplemented: tensor with zero element"
      %1281 = arith.divf %cst_2, %in : f32
      linalg.yield %1281 : f32
    } -> tensor<1280xf32>
    %expanded_241 = tensor.expand_shape %257 [[0, 1, 2]] output_shape [1280, 1, 1] : tensor<1280xf32> into tensor<1280x1x1xf32>
    %expanded_242 = tensor.expand_shape %808 [[0, 1, 2]] output_shape [1280, 1, 1] : tensor<1280xf32> into tensor<1280x1x1xf32>
    %809 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%804, %expanded_241 : tensor<1x1280x7x7xf32>, tensor<1280x1x1xf32>) outs(%802 : tensor<1x1280x7x7xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.subf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x1280x7x7xf32>
    %810 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%809, %expanded_242 : tensor<1x1280x7x7xf32>, tensor<1280x1x1xf32>) outs(%802 : tensor<1x1280x7x7xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.mulf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x1280x7x7xf32>
    %expanded_243 = tensor.expand_shape %259 [[0, 1, 2]] output_shape [1280, 1, 1] : tensor<1280xf32> into tensor<1280x1x1xf32>
    %811 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%810, %expanded_243 : tensor<1x1280x7x7xf32>, tensor<1280x1x1xf32>) outs(%802 : tensor<1x1280x7x7xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.mulf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x1280x7x7xf32>
    %expanded_244 = tensor.expand_shape %260 [[0, 1, 2]] output_shape [1280, 1, 1] : tensor<1280xf32> into tensor<1280x1x1xf32>
    %812 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%811, %expanded_244 : tensor<1x1280x7x7xf32>, tensor<1280x1x1xf32>) outs(%802 : tensor<1x1280x7x7xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.addf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x1280x7x7xf32>
    %813 = linalg.generic {indexing_maps = [#map1, #map5, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%812, %277 : tensor<1x1280x7x7xf32>, tensor<f32>) outs(%802 : tensor<1x1280x7x7xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.cmpf ogt, %in, %in_246 : f32
      %1281 = arith.select %1280, %in, %in_246 : f32
      linalg.yield %1281 : f32
    } -> tensor<1x1280x7x7xf32>
    %814 = linalg.generic {indexing_maps = [#map5, #map1, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%279, %813 : tensor<f32>, tensor<1x1280x7x7xf32>) outs(%802 : tensor<1x1280x7x7xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.cmpf olt, %in, %in_246 : f32
      %1281 = arith.select %1280, %in, %in_246 : f32
      linalg.yield %1281 : f32
    } -> tensor<1x1280x7x7xf32>
    %815 = tensor.empty() : tensor<1x1280x1x1xf32>
    %816 = linalg.fill ins(%cst_1 : f32) outs(%815 : tensor<1x1280x1x1xf32>) -> tensor<1x1280x1x1xf32>
    %817 = linalg.generic {indexing_maps = [#map3, #map6], iterator_types = ["parallel", "parallel", "reduction", "reduction"]} ins(%814 : tensor<1x1280x7x7xf32>) outs(%816 : tensor<1x1280x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = arith.addf %in, %out : f32
      linalg.yield %1280 : f32
    } -> tensor<1x1280x1x1xf32>
    %818 = linalg.generic {indexing_maps = [#map7, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%817 : tensor<1x1280x1x1xf32>) outs(%815 : tensor<1x1280x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1280 = arith.divf %in, %cst_4 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x1280x1x1xf32>
    %collapsed_245 = tensor.collapse_shape %818 [[0], [1, 2, 3]] : tensor<1x1280x1x1xf32> into tensor<1x1280xf32>
    %819 = tensor.empty() : tensor<1280x1001xf32>
    %transposed = linalg.transpose ins(%261 : tensor<1001x1280xf32>) outs(%819 : tensor<1280x1001xf32>) permutation = [1, 0] 
    %820 = tensor.empty() : tensor<1x1001xf32>
    %821 = linalg.fill ins(%cst_1 : f32) outs(%820 : tensor<1x1001xf32>) -> tensor<1x1001xf32>
    %822 = linalg.matmul ins(%collapsed_245, %transposed : tensor<1x1280xf32>, tensor<1280x1001xf32>) outs(%821 : tensor<1x1001xf32>) -> tensor<1x1001xf32>
    %823 = linalg.generic {indexing_maps = [#map8, #map9], iterator_types = ["parallel", "parallel"]} ins(%822 : tensor<1x1001xf32>) outs(%820 : tensor<1x1001xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x1001xf32>
    %824 = linalg.generic {indexing_maps = [#map8, #map10, #map9], iterator_types = ["parallel", "parallel"]} ins(%823, %262 : tensor<1x1001xf32>, tensor<1001xf32>) outs(%820 : tensor<1x1001xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %1280 = arith.addf %in, %in_246 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x1001xf32>
    %825:351 = hal.tensor.barrier join(%824, %1, %2, %3, %4, %6, %7, %8, %9, %11, %12, %13, %14, %16, %17, %18, %19, %21, %22, %23, %24, %26, %27, %28, %29, %31, %32, %33, %34, %36, %37, %38, %39, %41, %42, %43, %44, %46, %47, %48, %49, %51, %52, %53, %54, %56, %57, %58, %59, %61, %62, %63, %64, %66, %67, %68, %69, %71, %72, %73, %74, %76, %77, %78, %79, %81, %82, %83, %84, %86, %87, %88, %89, %91, %92, %93, %94, %96, %97, %98, %99, %101, %102, %103, %104, %106, %107, %108, %109, %111, %112, %113, %114, %116, %117, %118, %119, %121, %122, %123, %124, %126, %127, %128, %129, %131, %132, %133, %134, %136, %137, %138, %139, %141, %142, %143, %144, %146, %147, %148, %149, %151, %152, %153, %154, %156, %157, %158, %159, %161, %162, %163, %164, %166, %167, %168, %169, %171, %172, %173, %174, %176, %177, %178, %179, %181, %182, %183, %184, %186, %187, %188, %189, %191, %192, %193, %194, %196, %197, %198, %199, %201, %202, %203, %204, %206, %207, %208, %209, %211, %212, %213, %214, %216, %217, %218, %219, %221, %222, %223, %224, %226, %227, %228, %229, %231, %232, %233, %234, %236, %237, %238, %239, %241, %242, %243, %244, %246, %247, %248, %249, %251, %252, %253, %254, %256, %257, %258, %259, %padded, %265, %275, %271, %padded_8, %281, %288, %290, %293, %301, %304, %312, %padded_21, %317, %324, %326, %329, %337, %340, %348, %padded_35, %351, %358, %360, %361, %369, %370, %377, %padded_49, %382, %389, %391, %394, %401, %404, %412, %padded_63, %415, %422, %424, %425, %433, %434, %441, %padded_77, %444, %451, %453, %454, %462, %463, %470, %padded_91, %475, %482, %484, %487, %495, %498, %506, %padded_105, %509, %516, %518, %519, %527, %528, %535, %padded_119, %538, %545, %547, %548, %556, %557, %564, %padded_133, %567, %574, %576, %577, %585, %586, %593, %padded_147, %596, %603, %605, %608, %615, %618, %626, %padded_161, %629, %636, %638, %639, %647, %648, %655, %padded_175, %658, %665, %667, %668, %676, %677, %684, %padded_189, %689, %696, %698, %701, %709, %712, %720, %padded_203, %723, %730, %732, %733, %741, %742, %749, %padded_217, %752, %759, %761, %762, %770, %771, %778, %padded_231, %781, %788, %790, %793, %801, %804, %812, %collapsed_245, %transposed : tensor<1x1001xf32>, tensor<32x3x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x1x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<16x32x1x1xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<96x16x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x3x3xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<24x96x1x1xf32>, tensor<24xf32>, tensor<24xf32>, tensor<24xf32>, tensor<144x24x1x1xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144x1x3x3xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144xf32>, tensor<24x144x1x1xf32>, tensor<24xf32>, tensor<24xf32>, tensor<24xf32>, tensor<144x24x1x1xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144x1x3x3xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144xf32>, tensor<32x144x1x1xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<192x32x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x1x3x3xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<32x192x1x1xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<192x32x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x1x3x3xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<32x192x1x1xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<192x32x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x1x3x3xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<64x192x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<384x64x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x3x3xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<64x384x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<384x64x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x3x3xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<64x384x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<384x64x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x3x3xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<64x384x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<384x64x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x3x3xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<96x384x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<576x96x1x1xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576x1x3x3xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576xf32>, tensor<96x576x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<576x96x1x1xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576x1x3x3xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576xf32>, tensor<96x576x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<576x96x1x1xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576x1x3x3xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576xf32>, tensor<160x576x1x1xf32>, tensor<160xf32>, tensor<160xf32>, tensor<160xf32>, tensor<960x160x1x1xf32>, tensor<960xf32>, tensor<960xf32>, tensor<960xf32>, tensor<960x1x3x3xf32>, tensor<960xf32>, tensor<960xf32>, tensor<960xf32>, tensor<160x960x1x1xf32>, tensor<160xf32>, tensor<160xf32>, tensor<160xf32>, tensor<960x160x1x1xf32>, tensor<960xf32>, tensor<960xf32>, tensor<960xf32>, tensor<960x1x3x3xf32>, tensor<960xf32>, tensor<960xf32>, tensor<960xf32>, tensor<160x960x1x1xf32>, tensor<160xf32>, tensor<160xf32>, tensor<160xf32>, tensor<960x160x1x1xf32>, tensor<960xf32>, tensor<960xf32>, tensor<960xf32>, tensor<960x1x3x3xf32>, tensor<960xf32>, tensor<960xf32>, tensor<960xf32>, tensor<320x960x1x1xf32>, tensor<320xf32>, tensor<320xf32>, tensor<320xf32>, tensor<1280x320x1x1xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<1x3x225x225xf32>, tensor<1x32x112x112xf32>, tensor<1x32x112x112xf32>, tensor<0xf32>, tensor<1x32x114x114xf32>, tensor<1x32x112x112xf32>, tensor<1x32x112x112xf32>, tensor<1x32x112x112xf32>, tensor<1x16x112x112xf32>, tensor<1x16x112x112xf32>, tensor<1x96x112x112xf32>, tensor<1x96x112x112xf32>, tensor<1x96x113x113xf32>, tensor<1x96x56x56xf32>, tensor<1x96x56x56xf32>, tensor<1x96x56x56xf32>, tensor<1x24x56x56xf32>, tensor<1x24x56x56xf32>, tensor<1x144x56x56xf32>, tensor<1x144x56x56xf32>, tensor<1x144x58x58xf32>, tensor<1x144x56x56xf32>, tensor<1x144x56x56xf32>, tensor<1x144x56x56xf32>, tensor<1x24x56x56xf32>, tensor<1x24x56x56xf32>, tensor<1x144x56x56xf32>, tensor<1x144x56x56xf32>, tensor<1x144x57x57xf32>, tensor<1x144x28x28xf32>, tensor<1x144x28x28xf32>, tensor<1x144x28x28xf32>, tensor<1x32x28x28xf32>, tensor<1x32x28x28xf32>, tensor<1x192x28x28xf32>, tensor<1x192x28x28xf32>, tensor<1x192x30x30xf32>, tensor<1x192x28x28xf32>, tensor<1x192x28x28xf32>, tensor<1x192x28x28xf32>, tensor<1x32x28x28xf32>, tensor<1x32x28x28xf32>, tensor<1x192x28x28xf32>, tensor<1x192x28x28xf32>, tensor<1x192x30x30xf32>, tensor<1x192x28x28xf32>, tensor<1x192x28x28xf32>, tensor<1x192x28x28xf32>, tensor<1x32x28x28xf32>, tensor<1x32x28x28xf32>, tensor<1x192x28x28xf32>, tensor<1x192x28x28xf32>, tensor<1x192x29x29xf32>, tensor<1x192x14x14xf32>, tensor<1x192x14x14xf32>, tensor<1x192x14x14xf32>, tensor<1x64x14x14xf32>, tensor<1x64x14x14xf32>, tensor<1x384x14x14xf32>, tensor<1x384x14x14xf32>, tensor<1x384x16x16xf32>, tensor<1x384x14x14xf32>, tensor<1x384x14x14xf32>, tensor<1x384x14x14xf32>, tensor<1x64x14x14xf32>, tensor<1x64x14x14xf32>, tensor<1x384x14x14xf32>, tensor<1x384x14x14xf32>, tensor<1x384x16x16xf32>, tensor<1x384x14x14xf32>, tensor<1x384x14x14xf32>, tensor<1x384x14x14xf32>, tensor<1x64x14x14xf32>, tensor<1x64x14x14xf32>, tensor<1x384x14x14xf32>, tensor<1x384x14x14xf32>, tensor<1x384x16x16xf32>, tensor<1x384x14x14xf32>, tensor<1x384x14x14xf32>, tensor<1x384x14x14xf32>, tensor<1x64x14x14xf32>, tensor<1x64x14x14xf32>, tensor<1x384x14x14xf32>, tensor<1x384x14x14xf32>, tensor<1x384x16x16xf32>, tensor<1x384x14x14xf32>, tensor<1x384x14x14xf32>, tensor<1x384x14x14xf32>, tensor<1x96x14x14xf32>, tensor<1x96x14x14xf32>, tensor<1x576x14x14xf32>, tensor<1x576x14x14xf32>, tensor<1x576x16x16xf32>, tensor<1x576x14x14xf32>, tensor<1x576x14x14xf32>, tensor<1x576x14x14xf32>, tensor<1x96x14x14xf32>, tensor<1x96x14x14xf32>, tensor<1x576x14x14xf32>, tensor<1x576x14x14xf32>, tensor<1x576x16x16xf32>, tensor<1x576x14x14xf32>, tensor<1x576x14x14xf32>, tensor<1x576x14x14xf32>, tensor<1x96x14x14xf32>, tensor<1x96x14x14xf32>, tensor<1x576x14x14xf32>, tensor<1x576x14x14xf32>, tensor<1x576x15x15xf32>, tensor<1x576x7x7xf32>, tensor<1x576x7x7xf32>, tensor<1x576x7x7xf32>, tensor<1x160x7x7xf32>, tensor<1x160x7x7xf32>, tensor<1x960x7x7xf32>, tensor<1x960x7x7xf32>, tensor<1x960x9x9xf32>, tensor<1x960x7x7xf32>, tensor<1x960x7x7xf32>, tensor<1x960x7x7xf32>, tensor<1x160x7x7xf32>, tensor<1x160x7x7xf32>, tensor<1x960x7x7xf32>, tensor<1x960x7x7xf32>, tensor<1x960x9x9xf32>, tensor<1x960x7x7xf32>, tensor<1x960x7x7xf32>, tensor<1x960x7x7xf32>, tensor<1x160x7x7xf32>, tensor<1x160x7x7xf32>, tensor<1x960x7x7xf32>, tensor<1x960x7x7xf32>, tensor<1x960x9x9xf32>, tensor<1x960x7x7xf32>, tensor<1x960x7x7xf32>, tensor<1x960x7x7xf32>, tensor<1x320x7x7xf32>, tensor<1x320x7x7xf32>, tensor<1x1280x7x7xf32>, tensor<1x1280x7x7xf32>, tensor<1x1280xf32>, tensor<1280x1001xf32>) => %arg264 : !hal.fence
    %826 = hal.tensor.export %825#0 : tensor<1x1001xf32> -> !hal.buffer_view
    %827 = hal.tensor.export %825#1 : tensor<32x3x3x3xf32> -> !hal.buffer_view
    %828 = hal.tensor.export %825#2 : tensor<32xf32> -> !hal.buffer_view
    %829 = hal.tensor.export %825#3 : tensor<32xf32> -> !hal.buffer_view
    %830 = hal.tensor.export %825#4 : tensor<32xf32> -> !hal.buffer_view
    %831 = hal.tensor.export %825#5 : tensor<32x1x3x3xf32> -> !hal.buffer_view
    %832 = hal.tensor.export %825#6 : tensor<32xf32> -> !hal.buffer_view
    %833 = hal.tensor.export %825#7 : tensor<32xf32> -> !hal.buffer_view
    %834 = hal.tensor.export %825#8 : tensor<32xf32> -> !hal.buffer_view
    %835 = hal.tensor.export %825#9 : tensor<16x32x1x1xf32> -> !hal.buffer_view
    %836 = hal.tensor.export %825#10 : tensor<16xf32> -> !hal.buffer_view
    %837 = hal.tensor.export %825#11 : tensor<16xf32> -> !hal.buffer_view
    %838 = hal.tensor.export %825#12 : tensor<16xf32> -> !hal.buffer_view
    %839 = hal.tensor.export %825#13 : tensor<96x16x1x1xf32> -> !hal.buffer_view
    %840 = hal.tensor.export %825#14 : tensor<96xf32> -> !hal.buffer_view
    %841 = hal.tensor.export %825#15 : tensor<96xf32> -> !hal.buffer_view
    %842 = hal.tensor.export %825#16 : tensor<96xf32> -> !hal.buffer_view
    %843 = hal.tensor.export %825#17 : tensor<96x1x3x3xf32> -> !hal.buffer_view
    %844 = hal.tensor.export %825#18 : tensor<96xf32> -> !hal.buffer_view
    %845 = hal.tensor.export %825#19 : tensor<96xf32> -> !hal.buffer_view
    %846 = hal.tensor.export %825#20 : tensor<96xf32> -> !hal.buffer_view
    %847 = hal.tensor.export %825#21 : tensor<24x96x1x1xf32> -> !hal.buffer_view
    %848 = hal.tensor.export %825#22 : tensor<24xf32> -> !hal.buffer_view
    %849 = hal.tensor.export %825#23 : tensor<24xf32> -> !hal.buffer_view
    %850 = hal.tensor.export %825#24 : tensor<24xf32> -> !hal.buffer_view
    %851 = hal.tensor.export %825#25 : tensor<144x24x1x1xf32> -> !hal.buffer_view
    %852 = hal.tensor.export %825#26 : tensor<144xf32> -> !hal.buffer_view
    %853 = hal.tensor.export %825#27 : tensor<144xf32> -> !hal.buffer_view
    %854 = hal.tensor.export %825#28 : tensor<144xf32> -> !hal.buffer_view
    %855 = hal.tensor.export %825#29 : tensor<144x1x3x3xf32> -> !hal.buffer_view
    %856 = hal.tensor.export %825#30 : tensor<144xf32> -> !hal.buffer_view
    %857 = hal.tensor.export %825#31 : tensor<144xf32> -> !hal.buffer_view
    %858 = hal.tensor.export %825#32 : tensor<144xf32> -> !hal.buffer_view
    %859 = hal.tensor.export %825#33 : tensor<24x144x1x1xf32> -> !hal.buffer_view
    %860 = hal.tensor.export %825#34 : tensor<24xf32> -> !hal.buffer_view
    %861 = hal.tensor.export %825#35 : tensor<24xf32> -> !hal.buffer_view
    %862 = hal.tensor.export %825#36 : tensor<24xf32> -> !hal.buffer_view
    %863 = hal.tensor.export %825#37 : tensor<144x24x1x1xf32> -> !hal.buffer_view
    %864 = hal.tensor.export %825#38 : tensor<144xf32> -> !hal.buffer_view
    %865 = hal.tensor.export %825#39 : tensor<144xf32> -> !hal.buffer_view
    %866 = hal.tensor.export %825#40 : tensor<144xf32> -> !hal.buffer_view
    %867 = hal.tensor.export %825#41 : tensor<144x1x3x3xf32> -> !hal.buffer_view
    %868 = hal.tensor.export %825#42 : tensor<144xf32> -> !hal.buffer_view
    %869 = hal.tensor.export %825#43 : tensor<144xf32> -> !hal.buffer_view
    %870 = hal.tensor.export %825#44 : tensor<144xf32> -> !hal.buffer_view
    %871 = hal.tensor.export %825#45 : tensor<32x144x1x1xf32> -> !hal.buffer_view
    %872 = hal.tensor.export %825#46 : tensor<32xf32> -> !hal.buffer_view
    %873 = hal.tensor.export %825#47 : tensor<32xf32> -> !hal.buffer_view
    %874 = hal.tensor.export %825#48 : tensor<32xf32> -> !hal.buffer_view
    %875 = hal.tensor.export %825#49 : tensor<192x32x1x1xf32> -> !hal.buffer_view
    %876 = hal.tensor.export %825#50 : tensor<192xf32> -> !hal.buffer_view
    %877 = hal.tensor.export %825#51 : tensor<192xf32> -> !hal.buffer_view
    %878 = hal.tensor.export %825#52 : tensor<192xf32> -> !hal.buffer_view
    %879 = hal.tensor.export %825#53 : tensor<192x1x3x3xf32> -> !hal.buffer_view
    %880 = hal.tensor.export %825#54 : tensor<192xf32> -> !hal.buffer_view
    %881 = hal.tensor.export %825#55 : tensor<192xf32> -> !hal.buffer_view
    %882 = hal.tensor.export %825#56 : tensor<192xf32> -> !hal.buffer_view
    %883 = hal.tensor.export %825#57 : tensor<32x192x1x1xf32> -> !hal.buffer_view
    %884 = hal.tensor.export %825#58 : tensor<32xf32> -> !hal.buffer_view
    %885 = hal.tensor.export %825#59 : tensor<32xf32> -> !hal.buffer_view
    %886 = hal.tensor.export %825#60 : tensor<32xf32> -> !hal.buffer_view
    %887 = hal.tensor.export %825#61 : tensor<192x32x1x1xf32> -> !hal.buffer_view
    %888 = hal.tensor.export %825#62 : tensor<192xf32> -> !hal.buffer_view
    %889 = hal.tensor.export %825#63 : tensor<192xf32> -> !hal.buffer_view
    %890 = hal.tensor.export %825#64 : tensor<192xf32> -> !hal.buffer_view
    %891 = hal.tensor.export %825#65 : tensor<192x1x3x3xf32> -> !hal.buffer_view
    %892 = hal.tensor.export %825#66 : tensor<192xf32> -> !hal.buffer_view
    %893 = hal.tensor.export %825#67 : tensor<192xf32> -> !hal.buffer_view
    %894 = hal.tensor.export %825#68 : tensor<192xf32> -> !hal.buffer_view
    %895 = hal.tensor.export %825#69 : tensor<32x192x1x1xf32> -> !hal.buffer_view
    %896 = hal.tensor.export %825#70 : tensor<32xf32> -> !hal.buffer_view
    %897 = hal.tensor.export %825#71 : tensor<32xf32> -> !hal.buffer_view
    %898 = hal.tensor.export %825#72 : tensor<32xf32> -> !hal.buffer_view
    %899 = hal.tensor.export %825#73 : tensor<192x32x1x1xf32> -> !hal.buffer_view
    %900 = hal.tensor.export %825#74 : tensor<192xf32> -> !hal.buffer_view
    %901 = hal.tensor.export %825#75 : tensor<192xf32> -> !hal.buffer_view
    %902 = hal.tensor.export %825#76 : tensor<192xf32> -> !hal.buffer_view
    %903 = hal.tensor.export %825#77 : tensor<192x1x3x3xf32> -> !hal.buffer_view
    %904 = hal.tensor.export %825#78 : tensor<192xf32> -> !hal.buffer_view
    %905 = hal.tensor.export %825#79 : tensor<192xf32> -> !hal.buffer_view
    %906 = hal.tensor.export %825#80 : tensor<192xf32> -> !hal.buffer_view
    %907 = hal.tensor.export %825#81 : tensor<64x192x1x1xf32> -> !hal.buffer_view
    %908 = hal.tensor.export %825#82 : tensor<64xf32> -> !hal.buffer_view
    %909 = hal.tensor.export %825#83 : tensor<64xf32> -> !hal.buffer_view
    %910 = hal.tensor.export %825#84 : tensor<64xf32> -> !hal.buffer_view
    %911 = hal.tensor.export %825#85 : tensor<384x64x1x1xf32> -> !hal.buffer_view
    %912 = hal.tensor.export %825#86 : tensor<384xf32> -> !hal.buffer_view
    %913 = hal.tensor.export %825#87 : tensor<384xf32> -> !hal.buffer_view
    %914 = hal.tensor.export %825#88 : tensor<384xf32> -> !hal.buffer_view
    %915 = hal.tensor.export %825#89 : tensor<384x1x3x3xf32> -> !hal.buffer_view
    %916 = hal.tensor.export %825#90 : tensor<384xf32> -> !hal.buffer_view
    %917 = hal.tensor.export %825#91 : tensor<384xf32> -> !hal.buffer_view
    %918 = hal.tensor.export %825#92 : tensor<384xf32> -> !hal.buffer_view
    %919 = hal.tensor.export %825#93 : tensor<64x384x1x1xf32> -> !hal.buffer_view
    %920 = hal.tensor.export %825#94 : tensor<64xf32> -> !hal.buffer_view
    %921 = hal.tensor.export %825#95 : tensor<64xf32> -> !hal.buffer_view
    %922 = hal.tensor.export %825#96 : tensor<64xf32> -> !hal.buffer_view
    %923 = hal.tensor.export %825#97 : tensor<384x64x1x1xf32> -> !hal.buffer_view
    %924 = hal.tensor.export %825#98 : tensor<384xf32> -> !hal.buffer_view
    %925 = hal.tensor.export %825#99 : tensor<384xf32> -> !hal.buffer_view
    %926 = hal.tensor.export %825#100 : tensor<384xf32> -> !hal.buffer_view
    %927 = hal.tensor.export %825#101 : tensor<384x1x3x3xf32> -> !hal.buffer_view
    %928 = hal.tensor.export %825#102 : tensor<384xf32> -> !hal.buffer_view
    %929 = hal.tensor.export %825#103 : tensor<384xf32> -> !hal.buffer_view
    %930 = hal.tensor.export %825#104 : tensor<384xf32> -> !hal.buffer_view
    %931 = hal.tensor.export %825#105 : tensor<64x384x1x1xf32> -> !hal.buffer_view
    %932 = hal.tensor.export %825#106 : tensor<64xf32> -> !hal.buffer_view
    %933 = hal.tensor.export %825#107 : tensor<64xf32> -> !hal.buffer_view
    %934 = hal.tensor.export %825#108 : tensor<64xf32> -> !hal.buffer_view
    %935 = hal.tensor.export %825#109 : tensor<384x64x1x1xf32> -> !hal.buffer_view
    %936 = hal.tensor.export %825#110 : tensor<384xf32> -> !hal.buffer_view
    %937 = hal.tensor.export %825#111 : tensor<384xf32> -> !hal.buffer_view
    %938 = hal.tensor.export %825#112 : tensor<384xf32> -> !hal.buffer_view
    %939 = hal.tensor.export %825#113 : tensor<384x1x3x3xf32> -> !hal.buffer_view
    %940 = hal.tensor.export %825#114 : tensor<384xf32> -> !hal.buffer_view
    %941 = hal.tensor.export %825#115 : tensor<384xf32> -> !hal.buffer_view
    %942 = hal.tensor.export %825#116 : tensor<384xf32> -> !hal.buffer_view
    %943 = hal.tensor.export %825#117 : tensor<64x384x1x1xf32> -> !hal.buffer_view
    %944 = hal.tensor.export %825#118 : tensor<64xf32> -> !hal.buffer_view
    %945 = hal.tensor.export %825#119 : tensor<64xf32> -> !hal.buffer_view
    %946 = hal.tensor.export %825#120 : tensor<64xf32> -> !hal.buffer_view
    %947 = hal.tensor.export %825#121 : tensor<384x64x1x1xf32> -> !hal.buffer_view
    %948 = hal.tensor.export %825#122 : tensor<384xf32> -> !hal.buffer_view
    %949 = hal.tensor.export %825#123 : tensor<384xf32> -> !hal.buffer_view
    %950 = hal.tensor.export %825#124 : tensor<384xf32> -> !hal.buffer_view
    %951 = hal.tensor.export %825#125 : tensor<384x1x3x3xf32> -> !hal.buffer_view
    %952 = hal.tensor.export %825#126 : tensor<384xf32> -> !hal.buffer_view
    %953 = hal.tensor.export %825#127 : tensor<384xf32> -> !hal.buffer_view
    %954 = hal.tensor.export %825#128 : tensor<384xf32> -> !hal.buffer_view
    %955 = hal.tensor.export %825#129 : tensor<96x384x1x1xf32> -> !hal.buffer_view
    %956 = hal.tensor.export %825#130 : tensor<96xf32> -> !hal.buffer_view
    %957 = hal.tensor.export %825#131 : tensor<96xf32> -> !hal.buffer_view
    %958 = hal.tensor.export %825#132 : tensor<96xf32> -> !hal.buffer_view
    %959 = hal.tensor.export %825#133 : tensor<576x96x1x1xf32> -> !hal.buffer_view
    %960 = hal.tensor.export %825#134 : tensor<576xf32> -> !hal.buffer_view
    %961 = hal.tensor.export %825#135 : tensor<576xf32> -> !hal.buffer_view
    %962 = hal.tensor.export %825#136 : tensor<576xf32> -> !hal.buffer_view
    %963 = hal.tensor.export %825#137 : tensor<576x1x3x3xf32> -> !hal.buffer_view
    %964 = hal.tensor.export %825#138 : tensor<576xf32> -> !hal.buffer_view
    %965 = hal.tensor.export %825#139 : tensor<576xf32> -> !hal.buffer_view
    %966 = hal.tensor.export %825#140 : tensor<576xf32> -> !hal.buffer_view
    %967 = hal.tensor.export %825#141 : tensor<96x576x1x1xf32> -> !hal.buffer_view
    %968 = hal.tensor.export %825#142 : tensor<96xf32> -> !hal.buffer_view
    %969 = hal.tensor.export %825#143 : tensor<96xf32> -> !hal.buffer_view
    %970 = hal.tensor.export %825#144 : tensor<96xf32> -> !hal.buffer_view
    %971 = hal.tensor.export %825#145 : tensor<576x96x1x1xf32> -> !hal.buffer_view
    %972 = hal.tensor.export %825#146 : tensor<576xf32> -> !hal.buffer_view
    %973 = hal.tensor.export %825#147 : tensor<576xf32> -> !hal.buffer_view
    %974 = hal.tensor.export %825#148 : tensor<576xf32> -> !hal.buffer_view
    %975 = hal.tensor.export %825#149 : tensor<576x1x3x3xf32> -> !hal.buffer_view
    %976 = hal.tensor.export %825#150 : tensor<576xf32> -> !hal.buffer_view
    %977 = hal.tensor.export %825#151 : tensor<576xf32> -> !hal.buffer_view
    %978 = hal.tensor.export %825#152 : tensor<576xf32> -> !hal.buffer_view
    %979 = hal.tensor.export %825#153 : tensor<96x576x1x1xf32> -> !hal.buffer_view
    %980 = hal.tensor.export %825#154 : tensor<96xf32> -> !hal.buffer_view
    %981 = hal.tensor.export %825#155 : tensor<96xf32> -> !hal.buffer_view
    %982 = hal.tensor.export %825#156 : tensor<96xf32> -> !hal.buffer_view
    %983 = hal.tensor.export %825#157 : tensor<576x96x1x1xf32> -> !hal.buffer_view
    %984 = hal.tensor.export %825#158 : tensor<576xf32> -> !hal.buffer_view
    %985 = hal.tensor.export %825#159 : tensor<576xf32> -> !hal.buffer_view
    %986 = hal.tensor.export %825#160 : tensor<576xf32> -> !hal.buffer_view
    %987 = hal.tensor.export %825#161 : tensor<576x1x3x3xf32> -> !hal.buffer_view
    %988 = hal.tensor.export %825#162 : tensor<576xf32> -> !hal.buffer_view
    %989 = hal.tensor.export %825#163 : tensor<576xf32> -> !hal.buffer_view
    %990 = hal.tensor.export %825#164 : tensor<576xf32> -> !hal.buffer_view
    %991 = hal.tensor.export %825#165 : tensor<160x576x1x1xf32> -> !hal.buffer_view
    %992 = hal.tensor.export %825#166 : tensor<160xf32> -> !hal.buffer_view
    %993 = hal.tensor.export %825#167 : tensor<160xf32> -> !hal.buffer_view
    %994 = hal.tensor.export %825#168 : tensor<160xf32> -> !hal.buffer_view
    %995 = hal.tensor.export %825#169 : tensor<960x160x1x1xf32> -> !hal.buffer_view
    %996 = hal.tensor.export %825#170 : tensor<960xf32> -> !hal.buffer_view
    %997 = hal.tensor.export %825#171 : tensor<960xf32> -> !hal.buffer_view
    %998 = hal.tensor.export %825#172 : tensor<960xf32> -> !hal.buffer_view
    %999 = hal.tensor.export %825#173 : tensor<960x1x3x3xf32> -> !hal.buffer_view
    %1000 = hal.tensor.export %825#174 : tensor<960xf32> -> !hal.buffer_view
    %1001 = hal.tensor.export %825#175 : tensor<960xf32> -> !hal.buffer_view
    %1002 = hal.tensor.export %825#176 : tensor<960xf32> -> !hal.buffer_view
    %1003 = hal.tensor.export %825#177 : tensor<160x960x1x1xf32> -> !hal.buffer_view
    %1004 = hal.tensor.export %825#178 : tensor<160xf32> -> !hal.buffer_view
    %1005 = hal.tensor.export %825#179 : tensor<160xf32> -> !hal.buffer_view
    %1006 = hal.tensor.export %825#180 : tensor<160xf32> -> !hal.buffer_view
    %1007 = hal.tensor.export %825#181 : tensor<960x160x1x1xf32> -> !hal.buffer_view
    %1008 = hal.tensor.export %825#182 : tensor<960xf32> -> !hal.buffer_view
    %1009 = hal.tensor.export %825#183 : tensor<960xf32> -> !hal.buffer_view
    %1010 = hal.tensor.export %825#184 : tensor<960xf32> -> !hal.buffer_view
    %1011 = hal.tensor.export %825#185 : tensor<960x1x3x3xf32> -> !hal.buffer_view
    %1012 = hal.tensor.export %825#186 : tensor<960xf32> -> !hal.buffer_view
    %1013 = hal.tensor.export %825#187 : tensor<960xf32> -> !hal.buffer_view
    %1014 = hal.tensor.export %825#188 : tensor<960xf32> -> !hal.buffer_view
    %1015 = hal.tensor.export %825#189 : tensor<160x960x1x1xf32> -> !hal.buffer_view
    %1016 = hal.tensor.export %825#190 : tensor<160xf32> -> !hal.buffer_view
    %1017 = hal.tensor.export %825#191 : tensor<160xf32> -> !hal.buffer_view
    %1018 = hal.tensor.export %825#192 : tensor<160xf32> -> !hal.buffer_view
    %1019 = hal.tensor.export %825#193 : tensor<960x160x1x1xf32> -> !hal.buffer_view
    %1020 = hal.tensor.export %825#194 : tensor<960xf32> -> !hal.buffer_view
    %1021 = hal.tensor.export %825#195 : tensor<960xf32> -> !hal.buffer_view
    %1022 = hal.tensor.export %825#196 : tensor<960xf32> -> !hal.buffer_view
    %1023 = hal.tensor.export %825#197 : tensor<960x1x3x3xf32> -> !hal.buffer_view
    %1024 = hal.tensor.export %825#198 : tensor<960xf32> -> !hal.buffer_view
    %1025 = hal.tensor.export %825#199 : tensor<960xf32> -> !hal.buffer_view
    %1026 = hal.tensor.export %825#200 : tensor<960xf32> -> !hal.buffer_view
    %1027 = hal.tensor.export %825#201 : tensor<320x960x1x1xf32> -> !hal.buffer_view
    %1028 = hal.tensor.export %825#202 : tensor<320xf32> -> !hal.buffer_view
    %1029 = hal.tensor.export %825#203 : tensor<320xf32> -> !hal.buffer_view
    %1030 = hal.tensor.export %825#204 : tensor<320xf32> -> !hal.buffer_view
    %1031 = hal.tensor.export %825#205 : tensor<1280x320x1x1xf32> -> !hal.buffer_view
    %1032 = hal.tensor.export %825#206 : tensor<1280xf32> -> !hal.buffer_view
    %1033 = hal.tensor.export %825#207 : tensor<1280xf32> -> !hal.buffer_view
    %1034 = hal.tensor.export %825#208 : tensor<1280xf32> -> !hal.buffer_view
    %1035 = hal.tensor.export %825#209 : tensor<1x3x225x225xf32> -> !hal.buffer_view
    %1036 = hal.tensor.export %825#210 : tensor<1x32x112x112xf32> -> !hal.buffer_view
    %1037 = hal.tensor.export %825#211 : tensor<1x32x112x112xf32> -> !hal.buffer_view
    %1038 = hal.tensor.export %825#212 : tensor<0xf32> -> !hal.buffer_view
    %1039 = hal.tensor.export %825#212 : tensor<0xf32> -> !hal.buffer_view
    %1040 = hal.tensor.export %825#213 : tensor<1x32x114x114xf32> -> !hal.buffer_view
    %1041 = hal.tensor.export %825#214 : tensor<1x32x112x112xf32> -> !hal.buffer_view
    %1042 = hal.tensor.export %825#215 : tensor<1x32x112x112xf32> -> !hal.buffer_view
    %1043 = hal.tensor.export %825#212 : tensor<0xf32> -> !hal.buffer_view
    %1044 = hal.tensor.export %825#212 : tensor<0xf32> -> !hal.buffer_view
    %1045 = hal.tensor.export %825#216 : tensor<1x32x112x112xf32> -> !hal.buffer_view
    %1046 = hal.tensor.export %825#217 : tensor<1x16x112x112xf32> -> !hal.buffer_view
    %1047 = hal.tensor.export %825#212 : tensor<0xf32> -> !hal.buffer_view
    %1048 = hal.tensor.export %825#212 : tensor<0xf32> -> !hal.buffer_view
    %1049 = hal.tensor.export %825#218 : tensor<1x16x112x112xf32> -> !hal.buffer_view
    %1050 = hal.tensor.export %825#219 : tensor<1x96x112x112xf32> -> !hal.buffer_view
    %1051 = hal.tensor.export %825#220 : tensor<1x96x112x112xf32> -> !hal.buffer_view
    %1052 = hal.tensor.export %825#212 : tensor<0xf32> -> !hal.buffer_view
    %1053 = hal.tensor.export %825#212 : tensor<0xf32> -> !hal.buffer_view
    %1054 = hal.tensor.export %825#221 : tensor<1x96x113x113xf32> -> !hal.buffer_view
    %1055 = hal.tensor.export %825#222 : tensor<1x96x56x56xf32> -> !hal.buffer_view
    %1056 = hal.tensor.export %825#223 : tensor<1x96x56x56xf32> -> !hal.buffer_view
    %1057 = hal.tensor.export %825#212 : tensor<0xf32> -> !hal.buffer_view
    %1058 = hal.tensor.export %825#212 : tensor<0xf32> -> !hal.buffer_view
    %1059 = hal.tensor.export %825#224 : tensor<1x96x56x56xf32> -> !hal.buffer_view
    %1060 = hal.tensor.export %825#225 : tensor<1x24x56x56xf32> -> !hal.buffer_view
    %1061 = hal.tensor.export %825#212 : tensor<0xf32> -> !hal.buffer_view
    %1062 = hal.tensor.export %825#212 : tensor<0xf32> -> !hal.buffer_view
    %1063 = hal.tensor.export %825#226 : tensor<1x24x56x56xf32> -> !hal.buffer_view
    %1064 = hal.tensor.export %825#227 : tensor<1x144x56x56xf32> -> !hal.buffer_view
    %1065 = hal.tensor.export %825#228 : tensor<1x144x56x56xf32> -> !hal.buffer_view
    %1066 = hal.tensor.export %825#212 : tensor<0xf32> -> !hal.buffer_view
    %1067 = hal.tensor.export %825#212 : tensor<0xf32> -> !hal.buffer_view
    %1068 = hal.tensor.export %825#229 : tensor<1x144x58x58xf32> -> !hal.buffer_view
    %1069 = hal.tensor.export %825#230 : tensor<1x144x56x56xf32> -> !hal.buffer_view
    %1070 = hal.tensor.export %825#231 : tensor<1x144x56x56xf32> -> !hal.buffer_view
    %1071 = hal.tensor.export %825#212 : tensor<0xf32> -> !hal.buffer_view
    %1072 = hal.tensor.export %825#212 : tensor<0xf32> -> !hal.buffer_view
    %1073 = hal.tensor.export %825#232 : tensor<1x144x56x56xf32> -> !hal.buffer_view
    %1074 = hal.tensor.export %825#233 : tensor<1x24x56x56xf32> -> !hal.buffer_view
    %1075 = hal.tensor.export %825#212 : tensor<0xf32> -> !hal.buffer_view
    %1076 = hal.tensor.export %825#212 : tensor<0xf32> -> !hal.buffer_view
    %1077 = hal.tensor.export %825#234 : tensor<1x24x56x56xf32> -> !hal.buffer_view
    %1078 = hal.tensor.export %825#235 : tensor<1x144x56x56xf32> -> !hal.buffer_view
    %1079 = hal.tensor.export %825#236 : tensor<1x144x56x56xf32> -> !hal.buffer_view
    %1080 = hal.tensor.export %825#212 : tensor<0xf32> -> !hal.buffer_view
    %1081 = hal.tensor.export %825#212 : tensor<0xf32> -> !hal.buffer_view
    %1082 = hal.tensor.export %825#237 : tensor<1x144x57x57xf32> -> !hal.buffer_view
    %1083 = hal.tensor.export %825#238 : tensor<1x144x28x28xf32> -> !hal.buffer_view
    %1084 = hal.tensor.export %825#239 : tensor<1x144x28x28xf32> -> !hal.buffer_view
    %1085 = hal.tensor.export %825#212 : tensor<0xf32> -> !hal.buffer_view
    %1086 = hal.tensor.export %825#212 : tensor<0xf32> -> !hal.buffer_view
    %1087 = hal.tensor.export %825#240 : tensor<1x144x28x28xf32> -> !hal.buffer_view
    %1088 = hal.tensor.export %825#241 : tensor<1x32x28x28xf32> -> !hal.buffer_view
    %1089 = hal.tensor.export %825#212 : tensor<0xf32> -> !hal.buffer_view
    %1090 = hal.tensor.export %825#212 : tensor<0xf32> -> !hal.buffer_view
    %1091 = hal.tensor.export %825#242 : tensor<1x32x28x28xf32> -> !hal.buffer_view
    %1092 = hal.tensor.export %825#243 : tensor<1x192x28x28xf32> -> !hal.buffer_view
    %1093 = hal.tensor.export %825#244 : tensor<1x192x28x28xf32> -> !hal.buffer_view
    %1094 = hal.tensor.export %825#212 : tensor<0xf32> -> !hal.buffer_view
    %1095 = hal.tensor.export %825#212 : tensor<0xf32> -> !hal.buffer_view
    %1096 = hal.tensor.export %825#245 : tensor<1x192x30x30xf32> -> !hal.buffer_view
    %1097 = hal.tensor.export %825#246 : tensor<1x192x28x28xf32> -> !hal.buffer_view
    %1098 = hal.tensor.export %825#247 : tensor<1x192x28x28xf32> -> !hal.buffer_view
    %1099 = hal.tensor.export %825#212 : tensor<0xf32> -> !hal.buffer_view
    %1100 = hal.tensor.export %825#212 : tensor<0xf32> -> !hal.buffer_view
    %1101 = hal.tensor.export %825#248 : tensor<1x192x28x28xf32> -> !hal.buffer_view
    %1102 = hal.tensor.export %825#249 : tensor<1x32x28x28xf32> -> !hal.buffer_view
    %1103 = hal.tensor.export %825#212 : tensor<0xf32> -> !hal.buffer_view
    %1104 = hal.tensor.export %825#212 : tensor<0xf32> -> !hal.buffer_view
    %1105 = hal.tensor.export %825#250 : tensor<1x32x28x28xf32> -> !hal.buffer_view
    %1106 = hal.tensor.export %825#251 : tensor<1x192x28x28xf32> -> !hal.buffer_view
    %1107 = hal.tensor.export %825#252 : tensor<1x192x28x28xf32> -> !hal.buffer_view
    %1108 = hal.tensor.export %825#212 : tensor<0xf32> -> !hal.buffer_view
    %1109 = hal.tensor.export %825#212 : tensor<0xf32> -> !hal.buffer_view
    %1110 = hal.tensor.export %825#253 : tensor<1x192x30x30xf32> -> !hal.buffer_view
    %1111 = hal.tensor.export %825#254 : tensor<1x192x28x28xf32> -> !hal.buffer_view
    %1112 = hal.tensor.export %825#255 : tensor<1x192x28x28xf32> -> !hal.buffer_view
    %1113 = hal.tensor.export %825#212 : tensor<0xf32> -> !hal.buffer_view
    %1114 = hal.tensor.export %825#212 : tensor<0xf32> -> !hal.buffer_view
    %1115 = hal.tensor.export %825#256 : tensor<1x192x28x28xf32> -> !hal.buffer_view
    %1116 = hal.tensor.export %825#257 : tensor<1x32x28x28xf32> -> !hal.buffer_view
    %1117 = hal.tensor.export %825#212 : tensor<0xf32> -> !hal.buffer_view
    %1118 = hal.tensor.export %825#212 : tensor<0xf32> -> !hal.buffer_view
    %1119 = hal.tensor.export %825#258 : tensor<1x32x28x28xf32> -> !hal.buffer_view
    %1120 = hal.tensor.export %825#259 : tensor<1x192x28x28xf32> -> !hal.buffer_view
    %1121 = hal.tensor.export %825#260 : tensor<1x192x28x28xf32> -> !hal.buffer_view
    %1122 = hal.tensor.export %825#212 : tensor<0xf32> -> !hal.buffer_view
    %1123 = hal.tensor.export %825#212 : tensor<0xf32> -> !hal.buffer_view
    %1124 = hal.tensor.export %825#261 : tensor<1x192x29x29xf32> -> !hal.buffer_view
    %1125 = hal.tensor.export %825#262 : tensor<1x192x14x14xf32> -> !hal.buffer_view
    %1126 = hal.tensor.export %825#263 : tensor<1x192x14x14xf32> -> !hal.buffer_view
    %1127 = hal.tensor.export %825#212 : tensor<0xf32> -> !hal.buffer_view
    %1128 = hal.tensor.export %825#212 : tensor<0xf32> -> !hal.buffer_view
    %1129 = hal.tensor.export %825#264 : tensor<1x192x14x14xf32> -> !hal.buffer_view
    %1130 = hal.tensor.export %825#265 : tensor<1x64x14x14xf32> -> !hal.buffer_view
    %1131 = hal.tensor.export %825#212 : tensor<0xf32> -> !hal.buffer_view
    %1132 = hal.tensor.export %825#212 : tensor<0xf32> -> !hal.buffer_view
    %1133 = hal.tensor.export %825#266 : tensor<1x64x14x14xf32> -> !hal.buffer_view
    %1134 = hal.tensor.export %825#267 : tensor<1x384x14x14xf32> -> !hal.buffer_view
    %1135 = hal.tensor.export %825#268 : tensor<1x384x14x14xf32> -> !hal.buffer_view
    %1136 = hal.tensor.export %825#212 : tensor<0xf32> -> !hal.buffer_view
    %1137 = hal.tensor.export %825#212 : tensor<0xf32> -> !hal.buffer_view
    %1138 = hal.tensor.export %825#269 : tensor<1x384x16x16xf32> -> !hal.buffer_view
    %1139 = hal.tensor.export %825#270 : tensor<1x384x14x14xf32> -> !hal.buffer_view
    %1140 = hal.tensor.export %825#271 : tensor<1x384x14x14xf32> -> !hal.buffer_view
    %1141 = hal.tensor.export %825#212 : tensor<0xf32> -> !hal.buffer_view
    %1142 = hal.tensor.export %825#212 : tensor<0xf32> -> !hal.buffer_view
    %1143 = hal.tensor.export %825#272 : tensor<1x384x14x14xf32> -> !hal.buffer_view
    %1144 = hal.tensor.export %825#273 : tensor<1x64x14x14xf32> -> !hal.buffer_view
    %1145 = hal.tensor.export %825#212 : tensor<0xf32> -> !hal.buffer_view
    %1146 = hal.tensor.export %825#212 : tensor<0xf32> -> !hal.buffer_view
    %1147 = hal.tensor.export %825#274 : tensor<1x64x14x14xf32> -> !hal.buffer_view
    %1148 = hal.tensor.export %825#275 : tensor<1x384x14x14xf32> -> !hal.buffer_view
    %1149 = hal.tensor.export %825#276 : tensor<1x384x14x14xf32> -> !hal.buffer_view
    %1150 = hal.tensor.export %825#212 : tensor<0xf32> -> !hal.buffer_view
    %1151 = hal.tensor.export %825#212 : tensor<0xf32> -> !hal.buffer_view
    %1152 = hal.tensor.export %825#277 : tensor<1x384x16x16xf32> -> !hal.buffer_view
    %1153 = hal.tensor.export %825#278 : tensor<1x384x14x14xf32> -> !hal.buffer_view
    %1154 = hal.tensor.export %825#279 : tensor<1x384x14x14xf32> -> !hal.buffer_view
    %1155 = hal.tensor.export %825#212 : tensor<0xf32> -> !hal.buffer_view
    %1156 = hal.tensor.export %825#212 : tensor<0xf32> -> !hal.buffer_view
    %1157 = hal.tensor.export %825#280 : tensor<1x384x14x14xf32> -> !hal.buffer_view
    %1158 = hal.tensor.export %825#281 : tensor<1x64x14x14xf32> -> !hal.buffer_view
    %1159 = hal.tensor.export %825#212 : tensor<0xf32> -> !hal.buffer_view
    %1160 = hal.tensor.export %825#212 : tensor<0xf32> -> !hal.buffer_view
    %1161 = hal.tensor.export %825#282 : tensor<1x64x14x14xf32> -> !hal.buffer_view
    %1162 = hal.tensor.export %825#283 : tensor<1x384x14x14xf32> -> !hal.buffer_view
    %1163 = hal.tensor.export %825#284 : tensor<1x384x14x14xf32> -> !hal.buffer_view
    %1164 = hal.tensor.export %825#212 : tensor<0xf32> -> !hal.buffer_view
    %1165 = hal.tensor.export %825#212 : tensor<0xf32> -> !hal.buffer_view
    %1166 = hal.tensor.export %825#285 : tensor<1x384x16x16xf32> -> !hal.buffer_view
    %1167 = hal.tensor.export %825#286 : tensor<1x384x14x14xf32> -> !hal.buffer_view
    %1168 = hal.tensor.export %825#287 : tensor<1x384x14x14xf32> -> !hal.buffer_view
    %1169 = hal.tensor.export %825#212 : tensor<0xf32> -> !hal.buffer_view
    %1170 = hal.tensor.export %825#212 : tensor<0xf32> -> !hal.buffer_view
    %1171 = hal.tensor.export %825#288 : tensor<1x384x14x14xf32> -> !hal.buffer_view
    %1172 = hal.tensor.export %825#289 : tensor<1x64x14x14xf32> -> !hal.buffer_view
    %1173 = hal.tensor.export %825#212 : tensor<0xf32> -> !hal.buffer_view
    %1174 = hal.tensor.export %825#212 : tensor<0xf32> -> !hal.buffer_view
    %1175 = hal.tensor.export %825#290 : tensor<1x64x14x14xf32> -> !hal.buffer_view
    %1176 = hal.tensor.export %825#291 : tensor<1x384x14x14xf32> -> !hal.buffer_view
    %1177 = hal.tensor.export %825#292 : tensor<1x384x14x14xf32> -> !hal.buffer_view
    %1178 = hal.tensor.export %825#212 : tensor<0xf32> -> !hal.buffer_view
    %1179 = hal.tensor.export %825#212 : tensor<0xf32> -> !hal.buffer_view
    %1180 = hal.tensor.export %825#293 : tensor<1x384x16x16xf32> -> !hal.buffer_view
    %1181 = hal.tensor.export %825#294 : tensor<1x384x14x14xf32> -> !hal.buffer_view
    %1182 = hal.tensor.export %825#295 : tensor<1x384x14x14xf32> -> !hal.buffer_view
    %1183 = hal.tensor.export %825#212 : tensor<0xf32> -> !hal.buffer_view
    %1184 = hal.tensor.export %825#212 : tensor<0xf32> -> !hal.buffer_view
    %1185 = hal.tensor.export %825#296 : tensor<1x384x14x14xf32> -> !hal.buffer_view
    %1186 = hal.tensor.export %825#297 : tensor<1x96x14x14xf32> -> !hal.buffer_view
    %1187 = hal.tensor.export %825#212 : tensor<0xf32> -> !hal.buffer_view
    %1188 = hal.tensor.export %825#212 : tensor<0xf32> -> !hal.buffer_view
    %1189 = hal.tensor.export %825#298 : tensor<1x96x14x14xf32> -> !hal.buffer_view
    %1190 = hal.tensor.export %825#299 : tensor<1x576x14x14xf32> -> !hal.buffer_view
    %1191 = hal.tensor.export %825#300 : tensor<1x576x14x14xf32> -> !hal.buffer_view
    %1192 = hal.tensor.export %825#212 : tensor<0xf32> -> !hal.buffer_view
    %1193 = hal.tensor.export %825#212 : tensor<0xf32> -> !hal.buffer_view
    %1194 = hal.tensor.export %825#301 : tensor<1x576x16x16xf32> -> !hal.buffer_view
    %1195 = hal.tensor.export %825#302 : tensor<1x576x14x14xf32> -> !hal.buffer_view
    %1196 = hal.tensor.export %825#303 : tensor<1x576x14x14xf32> -> !hal.buffer_view
    %1197 = hal.tensor.export %825#212 : tensor<0xf32> -> !hal.buffer_view
    %1198 = hal.tensor.export %825#212 : tensor<0xf32> -> !hal.buffer_view
    %1199 = hal.tensor.export %825#304 : tensor<1x576x14x14xf32> -> !hal.buffer_view
    %1200 = hal.tensor.export %825#305 : tensor<1x96x14x14xf32> -> !hal.buffer_view
    %1201 = hal.tensor.export %825#212 : tensor<0xf32> -> !hal.buffer_view
    %1202 = hal.tensor.export %825#212 : tensor<0xf32> -> !hal.buffer_view
    %1203 = hal.tensor.export %825#306 : tensor<1x96x14x14xf32> -> !hal.buffer_view
    %1204 = hal.tensor.export %825#307 : tensor<1x576x14x14xf32> -> !hal.buffer_view
    %1205 = hal.tensor.export %825#308 : tensor<1x576x14x14xf32> -> !hal.buffer_view
    %1206 = hal.tensor.export %825#212 : tensor<0xf32> -> !hal.buffer_view
    %1207 = hal.tensor.export %825#212 : tensor<0xf32> -> !hal.buffer_view
    %1208 = hal.tensor.export %825#309 : tensor<1x576x16x16xf32> -> !hal.buffer_view
    %1209 = hal.tensor.export %825#310 : tensor<1x576x14x14xf32> -> !hal.buffer_view
    %1210 = hal.tensor.export %825#311 : tensor<1x576x14x14xf32> -> !hal.buffer_view
    %1211 = hal.tensor.export %825#212 : tensor<0xf32> -> !hal.buffer_view
    %1212 = hal.tensor.export %825#212 : tensor<0xf32> -> !hal.buffer_view
    %1213 = hal.tensor.export %825#312 : tensor<1x576x14x14xf32> -> !hal.buffer_view
    %1214 = hal.tensor.export %825#313 : tensor<1x96x14x14xf32> -> !hal.buffer_view
    %1215 = hal.tensor.export %825#212 : tensor<0xf32> -> !hal.buffer_view
    %1216 = hal.tensor.export %825#212 : tensor<0xf32> -> !hal.buffer_view
    %1217 = hal.tensor.export %825#314 : tensor<1x96x14x14xf32> -> !hal.buffer_view
    %1218 = hal.tensor.export %825#315 : tensor<1x576x14x14xf32> -> !hal.buffer_view
    %1219 = hal.tensor.export %825#316 : tensor<1x576x14x14xf32> -> !hal.buffer_view
    %1220 = hal.tensor.export %825#212 : tensor<0xf32> -> !hal.buffer_view
    %1221 = hal.tensor.export %825#212 : tensor<0xf32> -> !hal.buffer_view
    %1222 = hal.tensor.export %825#317 : tensor<1x576x15x15xf32> -> !hal.buffer_view
    %1223 = hal.tensor.export %825#318 : tensor<1x576x7x7xf32> -> !hal.buffer_view
    %1224 = hal.tensor.export %825#319 : tensor<1x576x7x7xf32> -> !hal.buffer_view
    %1225 = hal.tensor.export %825#212 : tensor<0xf32> -> !hal.buffer_view
    %1226 = hal.tensor.export %825#212 : tensor<0xf32> -> !hal.buffer_view
    %1227 = hal.tensor.export %825#320 : tensor<1x576x7x7xf32> -> !hal.buffer_view
    %1228 = hal.tensor.export %825#321 : tensor<1x160x7x7xf32> -> !hal.buffer_view
    %1229 = hal.tensor.export %825#212 : tensor<0xf32> -> !hal.buffer_view
    %1230 = hal.tensor.export %825#212 : tensor<0xf32> -> !hal.buffer_view
    %1231 = hal.tensor.export %825#322 : tensor<1x160x7x7xf32> -> !hal.buffer_view
    %1232 = hal.tensor.export %825#323 : tensor<1x960x7x7xf32> -> !hal.buffer_view
    %1233 = hal.tensor.export %825#324 : tensor<1x960x7x7xf32> -> !hal.buffer_view
    %1234 = hal.tensor.export %825#212 : tensor<0xf32> -> !hal.buffer_view
    %1235 = hal.tensor.export %825#212 : tensor<0xf32> -> !hal.buffer_view
    %1236 = hal.tensor.export %825#325 : tensor<1x960x9x9xf32> -> !hal.buffer_view
    %1237 = hal.tensor.export %825#326 : tensor<1x960x7x7xf32> -> !hal.buffer_view
    %1238 = hal.tensor.export %825#327 : tensor<1x960x7x7xf32> -> !hal.buffer_view
    %1239 = hal.tensor.export %825#212 : tensor<0xf32> -> !hal.buffer_view
    %1240 = hal.tensor.export %825#212 : tensor<0xf32> -> !hal.buffer_view
    %1241 = hal.tensor.export %825#328 : tensor<1x960x7x7xf32> -> !hal.buffer_view
    %1242 = hal.tensor.export %825#329 : tensor<1x160x7x7xf32> -> !hal.buffer_view
    %1243 = hal.tensor.export %825#212 : tensor<0xf32> -> !hal.buffer_view
    %1244 = hal.tensor.export %825#212 : tensor<0xf32> -> !hal.buffer_view
    %1245 = hal.tensor.export %825#330 : tensor<1x160x7x7xf32> -> !hal.buffer_view
    %1246 = hal.tensor.export %825#331 : tensor<1x960x7x7xf32> -> !hal.buffer_view
    %1247 = hal.tensor.export %825#332 : tensor<1x960x7x7xf32> -> !hal.buffer_view
    %1248 = hal.tensor.export %825#212 : tensor<0xf32> -> !hal.buffer_view
    %1249 = hal.tensor.export %825#212 : tensor<0xf32> -> !hal.buffer_view
    %1250 = hal.tensor.export %825#333 : tensor<1x960x9x9xf32> -> !hal.buffer_view
    %1251 = hal.tensor.export %825#334 : tensor<1x960x7x7xf32> -> !hal.buffer_view
    %1252 = hal.tensor.export %825#335 : tensor<1x960x7x7xf32> -> !hal.buffer_view
    %1253 = hal.tensor.export %825#212 : tensor<0xf32> -> !hal.buffer_view
    %1254 = hal.tensor.export %825#212 : tensor<0xf32> -> !hal.buffer_view
    %1255 = hal.tensor.export %825#336 : tensor<1x960x7x7xf32> -> !hal.buffer_view
    %1256 = hal.tensor.export %825#337 : tensor<1x160x7x7xf32> -> !hal.buffer_view
    %1257 = hal.tensor.export %825#212 : tensor<0xf32> -> !hal.buffer_view
    %1258 = hal.tensor.export %825#212 : tensor<0xf32> -> !hal.buffer_view
    %1259 = hal.tensor.export %825#338 : tensor<1x160x7x7xf32> -> !hal.buffer_view
    %1260 = hal.tensor.export %825#339 : tensor<1x960x7x7xf32> -> !hal.buffer_view
    %1261 = hal.tensor.export %825#340 : tensor<1x960x7x7xf32> -> !hal.buffer_view
    %1262 = hal.tensor.export %825#212 : tensor<0xf32> -> !hal.buffer_view
    %1263 = hal.tensor.export %825#212 : tensor<0xf32> -> !hal.buffer_view
    %1264 = hal.tensor.export %825#341 : tensor<1x960x9x9xf32> -> !hal.buffer_view
    %1265 = hal.tensor.export %825#342 : tensor<1x960x7x7xf32> -> !hal.buffer_view
    %1266 = hal.tensor.export %825#343 : tensor<1x960x7x7xf32> -> !hal.buffer_view
    %1267 = hal.tensor.export %825#212 : tensor<0xf32> -> !hal.buffer_view
    %1268 = hal.tensor.export %825#212 : tensor<0xf32> -> !hal.buffer_view
    %1269 = hal.tensor.export %825#344 : tensor<1x960x7x7xf32> -> !hal.buffer_view
    %1270 = hal.tensor.export %825#345 : tensor<1x320x7x7xf32> -> !hal.buffer_view
    %1271 = hal.tensor.export %825#212 : tensor<0xf32> -> !hal.buffer_view
    %1272 = hal.tensor.export %825#212 : tensor<0xf32> -> !hal.buffer_view
    %1273 = hal.tensor.export %825#346 : tensor<1x320x7x7xf32> -> !hal.buffer_view
    %1274 = hal.tensor.export %825#347 : tensor<1x1280x7x7xf32> -> !hal.buffer_view
    %1275 = hal.tensor.export %825#348 : tensor<1x1280x7x7xf32> -> !hal.buffer_view
    %1276 = hal.tensor.export %825#212 : tensor<0xf32> -> !hal.buffer_view
    %1277 = hal.tensor.export %825#212 : tensor<0xf32> -> !hal.buffer_view
    %1278 = hal.tensor.export %825#349 : tensor<1x1280xf32> -> !hal.buffer_view
    %1279 = hal.tensor.export %825#350 : tensor<1280x1001xf32> -> !hal.buffer_view
    util.return %826, %827, %828, %829, %830, %831, %832, %833, %834, %835, %836, %837, %838, %839, %840, %841, %842, %843, %844, %845, %846, %847, %848, %849, %850, %851, %852, %853, %854, %855, %856, %857, %858, %859, %860, %861, %862, %863, %864, %865, %866, %867, %868, %869, %870, %871, %872, %873, %874, %875, %876, %877, %878, %879, %880, %881, %882, %883, %884, %885, %886, %887, %888, %889, %890, %891, %892, %893, %894, %895, %896, %897, %898, %899, %900, %901, %902, %903, %904, %905, %906, %907, %908, %909, %910, %911, %912, %913, %914, %915, %916, %917, %918, %919, %920, %921, %922, %923, %924, %925, %926, %927, %928, %929, %930, %931, %932, %933, %934, %935, %936, %937, %938, %939, %940, %941, %942, %943, %944, %945, %946, %947, %948, %949, %950, %951, %952, %953, %954, %955, %956, %957, %958, %959, %960, %961, %962, %963, %964, %965, %966, %967, %968, %969, %970, %971, %972, %973, %974, %975, %976, %977, %978, %979, %980, %981, %982, %983, %984, %985, %986, %987, %988, %989, %990, %991, %992, %993, %994, %995, %996, %997, %998, %999, %1000, %1001, %1002, %1003, %1004, %1005, %1006, %1007, %1008, %1009, %1010, %1011, %1012, %1013, %1014, %1015, %1016, %1017, %1018, %1019, %1020, %1021, %1022, %1023, %1024, %1025, %1026, %1027, %1028, %1029, %1030, %1031, %1032, %1033, %1034, %1035, %1036, %1037, %1038, %1039, %1040, %1041, %1042, %1043, %1044, %1045, %1046, %1047, %1048, %1049, %1050, %1051, %1052, %1053, %1054, %1055, %1056, %1057, %1058, %1059, %1060, %1061, %1062, %1063, %1064, %1065, %1066, %1067, %1068, %1069, %1070, %1071, %1072, %1073, %1074, %1075, %1076, %1077, %1078, %1079, %1080, %1081, %1082, %1083, %1084, %1085, %1086, %1087, %1088, %1089, %1090, %1091, %1092, %1093, %1094, %1095, %1096, %1097, %1098, %1099, %1100, %1101, %1102, %1103, %1104, %1105, %1106, %1107, %1108, %1109, %1110, %1111, %1112, %1113, %1114, %1115, %1116, %1117, %1118, %1119, %1120, %1121, %1122, %1123, %1124, %1125, %1126, %1127, %1128, %1129, %1130, %1131, %1132, %1133, %1134, %1135, %1136, %1137, %1138, %1139, %1140, %1141, %1142, %1143, %1144, %1145, %1146, %1147, %1148, %1149, %1150, %1151, %1152, %1153, %1154, %1155, %1156, %1157, %1158, %1159, %1160, %1161, %1162, %1163, %1164, %1165, %1166, %1167, %1168, %1169, %1170, %1171, %1172, %1173, %1174, %1175, %1176, %1177, %1178, %1179, %1180, %1181, %1182, %1183, %1184, %1185, %1186, %1187, %1188, %1189, %1190, %1191, %1192, %1193, %1194, %1195, %1196, %1197, %1198, %1199, %1200, %1201, %1202, %1203, %1204, %1205, %1206, %1207, %1208, %1209, %1210, %1211, %1212, %1213, %1214, %1215, %1216, %1217, %1218, %1219, %1220, %1221, %1222, %1223, %1224, %1225, %1226, %1227, %1228, %1229, %1230, %1231, %1232, %1233, %1234, %1235, %1236, %1237, %1238, %1239, %1240, %1241, %1242, %1243, %1244, %1245, %1246, %1247, %1248, %1249, %1250, %1251, %1252, %1253, %1254, %1255, %1256, %1257, %1258, %1259, %1260, %1261, %1262, %1263, %1264, %1265, %1266, %1267, %1268, %1269, %1270, %1271, %1272, %1273, %1274, %1275, %1276, %1277, %1278, %1279 : !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view
  }
  util.func public @main(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view, %arg2: !hal.buffer_view, %arg3: !hal.buffer_view, %arg4: !hal.buffer_view, %arg5: !hal.buffer_view, %arg6: !hal.buffer_view, %arg7: !hal.buffer_view, %arg8: !hal.buffer_view, %arg9: !hal.buffer_view, %arg10: !hal.buffer_view, %arg11: !hal.buffer_view, %arg12: !hal.buffer_view, %arg13: !hal.buffer_view, %arg14: !hal.buffer_view, %arg15: !hal.buffer_view, %arg16: !hal.buffer_view, %arg17: !hal.buffer_view, %arg18: !hal.buffer_view, %arg19: !hal.buffer_view, %arg20: !hal.buffer_view, %arg21: !hal.buffer_view, %arg22: !hal.buffer_view, %arg23: !hal.buffer_view, %arg24: !hal.buffer_view, %arg25: !hal.buffer_view, %arg26: !hal.buffer_view, %arg27: !hal.buffer_view, %arg28: !hal.buffer_view, %arg29: !hal.buffer_view, %arg30: !hal.buffer_view, %arg31: !hal.buffer_view, %arg32: !hal.buffer_view, %arg33: !hal.buffer_view, %arg34: !hal.buffer_view, %arg35: !hal.buffer_view, %arg36: !hal.buffer_view, %arg37: !hal.buffer_view, %arg38: !hal.buffer_view, %arg39: !hal.buffer_view, %arg40: !hal.buffer_view, %arg41: !hal.buffer_view, %arg42: !hal.buffer_view, %arg43: !hal.buffer_view, %arg44: !hal.buffer_view, %arg45: !hal.buffer_view, %arg46: !hal.buffer_view, %arg47: !hal.buffer_view, %arg48: !hal.buffer_view, %arg49: !hal.buffer_view, %arg50: !hal.buffer_view, %arg51: !hal.buffer_view, %arg52: !hal.buffer_view, %arg53: !hal.buffer_view, %arg54: !hal.buffer_view, %arg55: !hal.buffer_view, %arg56: !hal.buffer_view, %arg57: !hal.buffer_view, %arg58: !hal.buffer_view, %arg59: !hal.buffer_view, %arg60: !hal.buffer_view, %arg61: !hal.buffer_view, %arg62: !hal.buffer_view, %arg63: !hal.buffer_view, %arg64: !hal.buffer_view, %arg65: !hal.buffer_view, %arg66: !hal.buffer_view, %arg67: !hal.buffer_view, %arg68: !hal.buffer_view, %arg69: !hal.buffer_view, %arg70: !hal.buffer_view, %arg71: !hal.buffer_view, %arg72: !hal.buffer_view, %arg73: !hal.buffer_view, %arg74: !hal.buffer_view, %arg75: !hal.buffer_view, %arg76: !hal.buffer_view, %arg77: !hal.buffer_view, %arg78: !hal.buffer_view, %arg79: !hal.buffer_view, %arg80: !hal.buffer_view, %arg81: !hal.buffer_view, %arg82: !hal.buffer_view, %arg83: !hal.buffer_view, %arg84: !hal.buffer_view, %arg85: !hal.buffer_view, %arg86: !hal.buffer_view, %arg87: !hal.buffer_view, %arg88: !hal.buffer_view, %arg89: !hal.buffer_view, %arg90: !hal.buffer_view, %arg91: !hal.buffer_view, %arg92: !hal.buffer_view, %arg93: !hal.buffer_view, %arg94: !hal.buffer_view, %arg95: !hal.buffer_view, %arg96: !hal.buffer_view, %arg97: !hal.buffer_view, %arg98: !hal.buffer_view, %arg99: !hal.buffer_view, %arg100: !hal.buffer_view, %arg101: !hal.buffer_view, %arg102: !hal.buffer_view, %arg103: !hal.buffer_view, %arg104: !hal.buffer_view, %arg105: !hal.buffer_view, %arg106: !hal.buffer_view, %arg107: !hal.buffer_view, %arg108: !hal.buffer_view, %arg109: !hal.buffer_view, %arg110: !hal.buffer_view, %arg111: !hal.buffer_view, %arg112: !hal.buffer_view, %arg113: !hal.buffer_view, %arg114: !hal.buffer_view, %arg115: !hal.buffer_view, %arg116: !hal.buffer_view, %arg117: !hal.buffer_view, %arg118: !hal.buffer_view, %arg119: !hal.buffer_view, %arg120: !hal.buffer_view, %arg121: !hal.buffer_view, %arg122: !hal.buffer_view, %arg123: !hal.buffer_view, %arg124: !hal.buffer_view, %arg125: !hal.buffer_view, %arg126: !hal.buffer_view, %arg127: !hal.buffer_view, %arg128: !hal.buffer_view, %arg129: !hal.buffer_view, %arg130: !hal.buffer_view, %arg131: !hal.buffer_view, %arg132: !hal.buffer_view, %arg133: !hal.buffer_view, %arg134: !hal.buffer_view, %arg135: !hal.buffer_view, %arg136: !hal.buffer_view, %arg137: !hal.buffer_view, %arg138: !hal.buffer_view, %arg139: !hal.buffer_view, %arg140: !hal.buffer_view, %arg141: !hal.buffer_view, %arg142: !hal.buffer_view, %arg143: !hal.buffer_view, %arg144: !hal.buffer_view, %arg145: !hal.buffer_view, %arg146: !hal.buffer_view, %arg147: !hal.buffer_view, %arg148: !hal.buffer_view, %arg149: !hal.buffer_view, %arg150: !hal.buffer_view, %arg151: !hal.buffer_view, %arg152: !hal.buffer_view, %arg153: !hal.buffer_view, %arg154: !hal.buffer_view, %arg155: !hal.buffer_view, %arg156: !hal.buffer_view, %arg157: !hal.buffer_view, %arg158: !hal.buffer_view, %arg159: !hal.buffer_view, %arg160: !hal.buffer_view, %arg161: !hal.buffer_view, %arg162: !hal.buffer_view, %arg163: !hal.buffer_view, %arg164: !hal.buffer_view, %arg165: !hal.buffer_view, %arg166: !hal.buffer_view, %arg167: !hal.buffer_view, %arg168: !hal.buffer_view, %arg169: !hal.buffer_view, %arg170: !hal.buffer_view, %arg171: !hal.buffer_view, %arg172: !hal.buffer_view, %arg173: !hal.buffer_view, %arg174: !hal.buffer_view, %arg175: !hal.buffer_view, %arg176: !hal.buffer_view, %arg177: !hal.buffer_view, %arg178: !hal.buffer_view, %arg179: !hal.buffer_view, %arg180: !hal.buffer_view, %arg181: !hal.buffer_view, %arg182: !hal.buffer_view, %arg183: !hal.buffer_view, %arg184: !hal.buffer_view, %arg185: !hal.buffer_view, %arg186: !hal.buffer_view, %arg187: !hal.buffer_view, %arg188: !hal.buffer_view, %arg189: !hal.buffer_view, %arg190: !hal.buffer_view, %arg191: !hal.buffer_view, %arg192: !hal.buffer_view, %arg193: !hal.buffer_view, %arg194: !hal.buffer_view, %arg195: !hal.buffer_view, %arg196: !hal.buffer_view, %arg197: !hal.buffer_view, %arg198: !hal.buffer_view, %arg199: !hal.buffer_view, %arg200: !hal.buffer_view, %arg201: !hal.buffer_view, %arg202: !hal.buffer_view, %arg203: !hal.buffer_view, %arg204: !hal.buffer_view, %arg205: !hal.buffer_view, %arg206: !hal.buffer_view, %arg207: !hal.buffer_view, %arg208: !hal.buffer_view, %arg209: !hal.buffer_view, %arg210: !hal.buffer_view, %arg211: !hal.buffer_view, %arg212: !hal.buffer_view, %arg213: !hal.buffer_view, %arg214: !hal.buffer_view, %arg215: !hal.buffer_view, %arg216: !hal.buffer_view, %arg217: !hal.buffer_view, %arg218: !hal.buffer_view, %arg219: !hal.buffer_view, %arg220: !hal.buffer_view, %arg221: !hal.buffer_view, %arg222: !hal.buffer_view, %arg223: !hal.buffer_view, %arg224: !hal.buffer_view, %arg225: !hal.buffer_view, %arg226: !hal.buffer_view, %arg227: !hal.buffer_view, %arg228: !hal.buffer_view, %arg229: !hal.buffer_view, %arg230: !hal.buffer_view, %arg231: !hal.buffer_view, %arg232: !hal.buffer_view, %arg233: !hal.buffer_view, %arg234: !hal.buffer_view, %arg235: !hal.buffer_view, %arg236: !hal.buffer_view, %arg237: !hal.buffer_view, %arg238: !hal.buffer_view, %arg239: !hal.buffer_view, %arg240: !hal.buffer_view, %arg241: !hal.buffer_view, %arg242: !hal.buffer_view, %arg243: !hal.buffer_view, %arg244: !hal.buffer_view, %arg245: !hal.buffer_view, %arg246: !hal.buffer_view, %arg247: !hal.buffer_view, %arg248: !hal.buffer_view, %arg249: !hal.buffer_view, %arg250: !hal.buffer_view, %arg251: !hal.buffer_view, %arg252: !hal.buffer_view, %arg253: !hal.buffer_view, %arg254: !hal.buffer_view, %arg255: !hal.buffer_view, %arg256: !hal.buffer_view, %arg257: !hal.buffer_view, %arg258: !hal.buffer_view, %arg259: !hal.buffer_view, %arg260: !hal.buffer_view, %arg261: !hal.buffer_view, %arg262: !hal.buffer_view) -> (!hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view) attributes {iree.abi.stub} {
    %0 = util.null : !hal.fence
    %c-1_i32 = arith.constant -1 : i32
    %c0 = arith.constant 0 : index
    %device_0 = hal.devices.get %c0 : !hal.device
    %fence = hal.fence.create device(%device_0 : !hal.device) flags("None") : !hal.fence
    %1:454 = util.call @main$async(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15, %arg16, %arg17, %arg18, %arg19, %arg20, %arg21, %arg22, %arg23, %arg24, %arg25, %arg26, %arg27, %arg28, %arg29, %arg30, %arg31, %arg32, %arg33, %arg34, %arg35, %arg36, %arg37, %arg38, %arg39, %arg40, %arg41, %arg42, %arg43, %arg44, %arg45, %arg46, %arg47, %arg48, %arg49, %arg50, %arg51, %arg52, %arg53, %arg54, %arg55, %arg56, %arg57, %arg58, %arg59, %arg60, %arg61, %arg62, %arg63, %arg64, %arg65, %arg66, %arg67, %arg68, %arg69, %arg70, %arg71, %arg72, %arg73, %arg74, %arg75, %arg76, %arg77, %arg78, %arg79, %arg80, %arg81, %arg82, %arg83, %arg84, %arg85, %arg86, %arg87, %arg88, %arg89, %arg90, %arg91, %arg92, %arg93, %arg94, %arg95, %arg96, %arg97, %arg98, %arg99, %arg100, %arg101, %arg102, %arg103, %arg104, %arg105, %arg106, %arg107, %arg108, %arg109, %arg110, %arg111, %arg112, %arg113, %arg114, %arg115, %arg116, %arg117, %arg118, %arg119, %arg120, %arg121, %arg122, %arg123, %arg124, %arg125, %arg126, %arg127, %arg128, %arg129, %arg130, %arg131, %arg132, %arg133, %arg134, %arg135, %arg136, %arg137, %arg138, %arg139, %arg140, %arg141, %arg142, %arg143, %arg144, %arg145, %arg146, %arg147, %arg148, %arg149, %arg150, %arg151, %arg152, %arg153, %arg154, %arg155, %arg156, %arg157, %arg158, %arg159, %arg160, %arg161, %arg162, %arg163, %arg164, %arg165, %arg166, %arg167, %arg168, %arg169, %arg170, %arg171, %arg172, %arg173, %arg174, %arg175, %arg176, %arg177, %arg178, %arg179, %arg180, %arg181, %arg182, %arg183, %arg184, %arg185, %arg186, %arg187, %arg188, %arg189, %arg190, %arg191, %arg192, %arg193, %arg194, %arg195, %arg196, %arg197, %arg198, %arg199, %arg200, %arg201, %arg202, %arg203, %arg204, %arg205, %arg206, %arg207, %arg208, %arg209, %arg210, %arg211, %arg212, %arg213, %arg214, %arg215, %arg216, %arg217, %arg218, %arg219, %arg220, %arg221, %arg222, %arg223, %arg224, %arg225, %arg226, %arg227, %arg228, %arg229, %arg230, %arg231, %arg232, %arg233, %arg234, %arg235, %arg236, %arg237, %arg238, %arg239, %arg240, %arg241, %arg242, %arg243, %arg244, %arg245, %arg246, %arg247, %arg248, %arg249, %arg250, %arg251, %arg252, %arg253, %arg254, %arg255, %arg256, %arg257, %arg258, %arg259, %arg260, %arg261, %arg262, %0, %fence) : (!hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.fence, !hal.fence) -> (!hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view)
    %status = hal.fence.await until([%fence]) timeout_millis(%c-1_i32) : i32
    util.return %1#0, %1#1, %1#2, %1#3, %1#4, %1#5, %1#6, %1#7, %1#8, %1#9, %1#10, %1#11, %1#12, %1#13, %1#14, %1#15, %1#16, %1#17, %1#18, %1#19, %1#20, %1#21, %1#22, %1#23, %1#24, %1#25, %1#26, %1#27, %1#28, %1#29, %1#30, %1#31, %1#32, %1#33, %1#34, %1#35, %1#36, %1#37, %1#38, %1#39, %1#40, %1#41, %1#42, %1#43, %1#44, %1#45, %1#46, %1#47, %1#48, %1#49, %1#50, %1#51, %1#52, %1#53, %1#54, %1#55, %1#56, %1#57, %1#58, %1#59, %1#60, %1#61, %1#62, %1#63, %1#64, %1#65, %1#66, %1#67, %1#68, %1#69, %1#70, %1#71, %1#72, %1#73, %1#74, %1#75, %1#76, %1#77, %1#78, %1#79, %1#80, %1#81, %1#82, %1#83, %1#84, %1#85, %1#86, %1#87, %1#88, %1#89, %1#90, %1#91, %1#92, %1#93, %1#94, %1#95, %1#96, %1#97, %1#98, %1#99, %1#100, %1#101, %1#102, %1#103, %1#104, %1#105, %1#106, %1#107, %1#108, %1#109, %1#110, %1#111, %1#112, %1#113, %1#114, %1#115, %1#116, %1#117, %1#118, %1#119, %1#120, %1#121, %1#122, %1#123, %1#124, %1#125, %1#126, %1#127, %1#128, %1#129, %1#130, %1#131, %1#132, %1#133, %1#134, %1#135, %1#136, %1#137, %1#138, %1#139, %1#140, %1#141, %1#142, %1#143, %1#144, %1#145, %1#146, %1#147, %1#148, %1#149, %1#150, %1#151, %1#152, %1#153, %1#154, %1#155, %1#156, %1#157, %1#158, %1#159, %1#160, %1#161, %1#162, %1#163, %1#164, %1#165, %1#166, %1#167, %1#168, %1#169, %1#170, %1#171, %1#172, %1#173, %1#174, %1#175, %1#176, %1#177, %1#178, %1#179, %1#180, %1#181, %1#182, %1#183, %1#184, %1#185, %1#186, %1#187, %1#188, %1#189, %1#190, %1#191, %1#192, %1#193, %1#194, %1#195, %1#196, %1#197, %1#198, %1#199, %1#200, %1#201, %1#202, %1#203, %1#204, %1#205, %1#206, %1#207, %1#208, %1#209, %1#210, %1#211, %1#212, %1#213, %1#214, %1#215, %1#216, %1#217, %1#218, %1#219, %1#220, %1#221, %1#222, %1#223, %1#224, %1#225, %1#226, %1#227, %1#228, %1#229, %1#230, %1#231, %1#232, %1#233, %1#234, %1#235, %1#236, %1#237, %1#238, %1#239, %1#240, %1#241, %1#242, %1#243, %1#244, %1#245, %1#246, %1#247, %1#248, %1#249, %1#250, %1#251, %1#252, %1#253, %1#254, %1#255, %1#256, %1#257, %1#258, %1#259, %1#260, %1#261, %1#262, %1#263, %1#264, %1#265, %1#266, %1#267, %1#268, %1#269, %1#270, %1#271, %1#272, %1#273, %1#274, %1#275, %1#276, %1#277, %1#278, %1#279, %1#280, %1#281, %1#282, %1#283, %1#284, %1#285, %1#286, %1#287, %1#288, %1#289, %1#290, %1#291, %1#292, %1#293, %1#294, %1#295, %1#296, %1#297, %1#298, %1#299, %1#300, %1#301, %1#302, %1#303, %1#304, %1#305, %1#306, %1#307, %1#308, %1#309, %1#310, %1#311, %1#312, %1#313, %1#314, %1#315, %1#316, %1#317, %1#318, %1#319, %1#320, %1#321, %1#322, %1#323, %1#324, %1#325, %1#326, %1#327, %1#328, %1#329, %1#330, %1#331, %1#332, %1#333, %1#334, %1#335, %1#336, %1#337, %1#338, %1#339, %1#340, %1#341, %1#342, %1#343, %1#344, %1#345, %1#346, %1#347, %1#348, %1#349, %1#350, %1#351, %1#352, %1#353, %1#354, %1#355, %1#356, %1#357, %1#358, %1#359, %1#360, %1#361, %1#362, %1#363, %1#364, %1#365, %1#366, %1#367, %1#368, %1#369, %1#370, %1#371, %1#372, %1#373, %1#374, %1#375, %1#376, %1#377, %1#378, %1#379, %1#380, %1#381, %1#382, %1#383, %1#384, %1#385, %1#386, %1#387, %1#388, %1#389, %1#390, %1#391, %1#392, %1#393, %1#394, %1#395, %1#396, %1#397, %1#398, %1#399, %1#400, %1#401, %1#402, %1#403, %1#404, %1#405, %1#406, %1#407, %1#408, %1#409, %1#410, %1#411, %1#412, %1#413, %1#414, %1#415, %1#416, %1#417, %1#418, %1#419, %1#420, %1#421, %1#422, %1#423, %1#424, %1#425, %1#426, %1#427, %1#428, %1#429, %1#430, %1#431, %1#432, %1#433, %1#434, %1#435, %1#436, %1#437, %1#438, %1#439, %1#440, %1#441, %1#442, %1#443, %1#444, %1#445, %1#446, %1#447, %1#448, %1#449, %1#450, %1#451, %1#452, %1#453 : !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view
  }
}

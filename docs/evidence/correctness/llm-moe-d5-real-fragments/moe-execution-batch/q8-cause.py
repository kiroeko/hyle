import json, math, struct
from pathlib import Path
f32=lambda x: struct.unpack('<f',struct.pack('<f',x))[0]
f16=lambda x: struct.unpack('<e',struct.pack('<e',x))[0]
def bf16(x):
    bits=struct.unpack('<I',struct.pack('<f',x))[0]
    bits += 0x7fff + ((bits >> 16) & 1)
    return struct.unpack('<f',struct.pack('<I', bits & 0xffff0000))[0]
def rounded(x):
    return math.floor(x+0.5) if x>=0 else math.ceil(x-0.5)
observations=[]
for replay in range(3):
    seed=3+14*replay
    x=[bf16(math.cos((i+31*seed)*.017)*.0625+((i+seed)%7)*.002-.006) for i in range(4096)]
    for e in range(9): x[e]=2+(8-e if replay==1 else e)/8
    if replay==2: x[0]=5
    row=1921
    col=(17*row+127)%4096
    coefficient=f32(.5+(row+127)%13/32)
    # Sparse Q8_0 weight is the block maximum: its stored q=127 and F16 d.
    stored_coefficient=f32(127*f16(f32(coefficient/127)))
    block=x[col//32*32:col//32*32+32]
    d=f32(max(abs(v) for v in block)/127)
    q=rounded(f32(x[col]*f32(1/d)))
    native=f32(f32(127*q)*f32(f16(f32(coefficient/127))*f16(d)))
    observations.append(dict(replay=replay,row=row,column=col,input=x[col],
        block_max=max(abs(v) for v in block),activation_quant_step=f16(d),
        decoded_weight_oracle=f32(x[col]*stored_coefficient),
        independent_q8_activation_prediction=native))
path=Path('.scratch/validation/d5-moe-execution/q8-cause.json')
path.write_text(json.dumps(observations,indent=2)+'\n',encoding='utf-8')
print(path.read_text(encoding='utf-8'))

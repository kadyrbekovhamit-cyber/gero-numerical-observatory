from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import json, subprocess, math, textwrap

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'output'; ASSETS=ROOT/'assets'
OUT.mkdir(exist_ok=True); ASSETS.mkdir(exist_ok=True)
W,H=1080,1920
INK='#153b35'; CREAM='#f4f1e7'; MINT='#b9e4ca'; MUTED='#9db7ab'; AMBER='#f5b565'
FONT='/System/Library/Fonts/Helvetica.ttc'
def f(n,bold=False):return ImageFont.truetype(FONT,n,index=1 if bold else 0)
def run(args):return subprocess.run(args,check=True,capture_output=True,text=True)
def duration(p):return float(run(['ffprobe','-v','error','-show_entries','format=duration','-of','default=noprint_wrappers=1:nokey=1',str(p)]).stdout)
scenes=[
 {'title':'Check the\nreference.','label':'A QUESTION FOR ML ENGINEERS','source':'S1–S4','speech':'Can a reference implementation lose numerical accuracy? Yes. GERO tests the mathematical contract as well as comparing the two outputs.','caption':'Compare the outputs.\nTest the contract.','kind':'intro'},
 {'title':'246 evaluations.','label':'ONE DOCUMENTED CPU SNAPSHOT','source':'S1','speech':'We ran one hundred twenty-three synthetic graphs with runtime optimizations off and on. Two hundred thirty-two evaluations passed. Fourteen divergences came from seven graphs.','caption':'123 graphs × 2 optimization settings','kind':'counts'},
 {'title':'14 out of 14.','label':'REPLAY THE EVIDENCE','source':'S2','speech':'All fourteen flagged archives reproduced. We also checked the seven graph formulas with eighty-digit decimal calculations.','caption':'Algorithmically independent check.\nSame project; not third-party validation.','kind':'replay'},
 {'title':'17 non-finite\noutputs.','label':'LAYER NORMALIZATION · FLOAT32','source':'S2','speech':'One layer normalization graph returned seventeen non-finite values out of sixty-eight in ONNX Runtime. The high-precision calculation was finite.','caption':'One graph · [4, 17] · large offset\nONNX Runtime 1.22.1 · CPU','kind':'layer'},
 {'title':'The reference\ncan fail, too.','label':'LOG SOFTMAX · FLOAT32','source':'S2','speech':'In other cases, the official reference evaluator produced non-finite values. A disagreement needs investigation before assigning blame.','caption':'LogSoftmax / random-022\nReference: 458 non-finite / 508 outputs','kind':'reference'},
 {'title':'Keep the\nversion boundary.','label':'WHAT THE EVIDENCE SUPPORTS','source':'S1–S3','speech':'These are version-specific observations. We have not established new bugs or regressions between versions. Quantize Linear is excluded from this benchmark.','caption':'ONNX 1.19.0 · ORT 1.22.1\nNumPy 2.2.6 · macOS arm64 CPU','kind':'scope'},
 {'title':'Inspect.\nDownload.\nReplay.','label':'GERO NUMERICAL OBSERVATORY','source':'S1–S3','speech':'Explore the tolerances, test graphs, and reproducible reports at gero dot uz.','caption':'gero.uz/stability\nSources and code in the description.','kind':'end'},
]
(ROOT/'script.md').write_text('# Narration — verified against sources.md\n\n'+'\n\n'.join(s['speech'] for s in scenes)+'\n')
(ROOT/'storyboard.json').write_text(json.dumps(scenes,indent=2)+'\n')
parts=[]; total=0; timings=[]
def write_lines(d,text,xy,size=56,fill=INK,bold=False,spacing=16):
    x,y=xy
    for line in text.split('\n'):
        assert d.textlength(line,font=f(size,bold))<=W-x-76,(line,size)
        d.text((x,y),line,font=f(size,bold),fill=fill);y+=size+spacing
    return y
for i,s in enumerate(scenes):
    dark=i in (1,3,5)
    bg,fg=(INK,CREAM) if dark else (CREAM,INK)
    im=Image.new('RGB',(W,H),bg);d=ImageDraw.Draw(im)
    d.rounded_rectangle((76,86,157,167),radius=20,fill=MINT if dark else INK)
    d.text((97,91),'g',font=f(60,True),fill=INK if dark else CREAM)
    d.text((181,108),'GERO / TECHNOLOGY PRODUCT',font=f(29,True),fill=fg)
    d.line((76,216,1004,216),fill=MUTED,width=2)
    write_lines(d,s['label'],(76,284),size=27,fill=MINT if dark else INK,bold=True)
    write_lines(d,s['title'],(76,363),size=89 if i!=1 else 83,fill=fg,bold=True,spacing=14)
    if s['kind']=='intro':
        for y,title,sub in [(765,'ONNX Runtime','Production implementation'),(1055,'ONNX Reference','Official evaluator')]:
            d.rounded_rectangle((76,y,1004,y+210),radius=24,outline=INK,width=3)
            write_lines(d,title,(114,y+39),55,bold=True)
            write_lines(d,sub,(114,y+118),35)
        d.line((540,988,540,1036),fill=INK,width=4)
        d.line((528,1000,540,988,552,1000),fill=INK,width=4)
        d.line((528,1024,540,1036,552,1024),fill=INK,width=4)
        write_lines(d,'Same graph. Same inputs.',(76,1327),44)
    elif s['kind']=='counts':
        d.text((76,630),'232',font=f(215,True),fill=MINT)
        write_lines(d,'PASSED ALL CONFIGURED CHECKS',(85,875),31,fill=MINT,bold=True)
        d.text((76,986),'14',font=f(205,True),fill=AMBER)
        write_lines(d,'DIVERGENCE OBSERVATIONS',(85,1223),31,fill=AMBER,bold=True)
        d.rounded_rectangle((76,1340,1004,1386),radius=10,fill=AMBER)
        d.rounded_rectangle((76,1340,76+round(928*232/246),1386),radius=10,fill=MINT)
        write_lines(d,'7 distinct graphs',(76,1427),44,fill=fg)
    elif s['kind']=='replay':
        for k,label in enumerate(['Graph + exact inputs','Output + environment','Source + SHA-256 manifest']):
            y=758+k*166
            d.rounded_rectangle((76,y,1004,y+132),radius=20,fill='#e2e8dc')
            d.line((109,y+62,125,y+78,153,y+44),fill=INK,width=7)
            write_lines(d,label,(181,y+43),37,bold=True)
        write_lines(d,'80-digit Decimal',(76,1330),67,bold=True)
        write_lines(d,'All 7 diagnostic results finite',(76,1430),39)
    elif s['kind']=='layer':
        for k in range(68):
            x=76+(k%17)*54; y=815+(k//17)*110
            d.rounded_rectangle((x,y,x+40,y+73),radius=10,fill=AMBER if k<17 else MINT)
        write_lines(d,'17 non-finite / 68 outputs',(76,1310),49,fill=fg,bold=True)
        write_lines(d,'Count diagram; grouped by status',(76,1400),31,fill=MUTED)
    elif s['kind']=='reference':
        write_lines(d,'458 / 508',(76,828),130,bold=True)
        write_lines(d,'non-finite reference outputs',(76,1015),46)
        d.rounded_rectangle((76,1160,1004,1202),radius=10,fill='#d0d6cc')
        d.rounded_rectangle((76,1160,76+round(928*458/508),1202),radius=10,fill='#ad6434')
        write_lines(d,'Decimal diagnostic: all finite',(76,1330),43,bold=True)
        write_lines(d,'ORT: within the chosen tolerance',(76,1420),39)
    elif s['kind']=='scope':
        for k,label in enumerate(['Pinned versions','Synthetic graphs','CPU execution','No novelty claim']):
            y=809+k*151
            d.line((76,y+108,1004,y+108),fill=MUTED,width=2)
            write_lines(d,f'0{k+1}',(76,y),36,fill=MINT,bold=True)
            write_lines(d,label,(190,y-3),49,fill=fg,bold=True)
    else:
        write_lines(d,'Mathematical contracts',(76,893),47,bold=True)
        write_lines(d,'Batch behavior\nProbability normalization\nEquivalent graphs',(76,1010),47,spacing=29)
        d.rounded_rectangle((76,1305,1004,1434),radius=25,fill=INK)
        write_lines(d,'gero.uz/stability',(119,1330),66,fill=CREAM,bold=True)
    d.rounded_rectangle((52,1550,1028,1734),radius=24,fill='#0b2420' if dark else '#e2e8dc')
    write_lines(d,s['caption'],(84,1585),size=34,fill=fg,spacing=18)
    d.text((76,1800),f'EVIDENCE {s["source"]}  ·  06 SEP 2026',font=f(27),fill=MUTED if dark else '#536d63')
    d.text((76,1845),'Synthetic narration · Xamit Kadirbekov / GERO',font=f(26),fill=MUTED if dark else '#536d63')
    d.text((937,1800),f'{i+1:02d}',font=f(34,True),fill=fg)
    frame=ASSETS/f'scene-{i+1:02d}.png';im.save(frame)
    textfile=ASSETS/f'scene-{i+1:02d}.txt';textfile.write_text(s['speech'])
    voice=ASSETS/f'scene-{i+1:02d}.aiff'
    run(['say','-v','Daniel','-r','160','-f',str(textfile),'-o',str(voice)])
    seconds=math.ceil((duration(voice)+.65)*30)/30
    segment=ASSETS/f'scene-{i+1:02d}.mp4'
    run(['ffmpeg','-hide_banner','-loglevel','error','-y','-loop','1','-framerate','30','-i',str(frame),'-i',str(voice),'-t',str(seconds),'-vf',f'fade=t=in:st=0:d=0.18,fade=t=out:st={seconds-.2}:d=0.2','-af','apad=pad_dur=1','-c:v','libx264','-preset','veryfast','-crf','20','-pix_fmt','yuv420p','-c:a','aac','-b:a','160k','-ar','48000',str(segment)])
    parts.append(segment);timings.append({'start':total,'end':total+seconds,**s});total+=seconds
    print(f'Scene {i+1}: {seconds:.2f}s',flush=True)
listing=ASSETS/'concat.txt';listing.write_text(''.join(f"file '{p}'\n" for p in parts))
run(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','concat','-safe','0','-i',str(listing),'-c','copy','-movflags','+faststart',str(OUT/'episode.mp4')])
(OUT/'timings.json').write_text(json.dumps(timings,indent=2)+'\n')
def stamp(t):
    ms=round(t*1000);return f'{ms//3600000:02}:{ms//60000%60:02}:{ms//1000%60:02},{ms%1000:03}'
srt=[];n=1
for s in timings:
    chunks=textwrap.wrap(s['speech'],width=76,break_long_words=False,break_on_hyphens=False)
    if len(chunks)>1 and len(chunks[-1].split())<4:chunks[-2:]=[' '.join(chunks[-2:])]
    weights=[len(c.split()) for c in chunks];t=s['start'];end=s['end']-.5
    for chunk,weight in zip(chunks,weights):
        next_t=t+(end-s['start'])*weight/sum(weights)
        srt.append(f'{n}\n{stamp(t)} --> {stamp(next_t)}\n'+textwrap.fill(chunk,width=44,break_on_hyphens=False)+'\n');t=next_t;n+=1
(OUT/'captions.en.srt').write_text('\n'.join(srt))
thumb=Image.open(ASSETS/'scene-01.png');thumb.save(OUT/'thumbnail.png')
sheet=Image.new('RGB',(4*270,2*480),'#dadfd6')
for i in range(7):sheet.paste(Image.open(ASSETS/f'scene-{i+1:02d}.png').resize((270,480)),((i%4)*270,(i//4)*480))
sheet.save(OUT/'contact-sheet.jpg')
print(f'Complete: {total:.2f}s',flush=True)

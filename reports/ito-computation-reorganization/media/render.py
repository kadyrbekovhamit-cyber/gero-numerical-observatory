"""Original causal animation; sequential frames then one-thread FFmpeg, no playback."""
import sys,json,math,subprocess,hashlib,re,html,argparse
from pathlib import Path
R=Path(__file__).resolve().parent
sys.path.insert(0,str(R.parent/'.runtime'))
from PIL import Image,ImageDraw,ImageFont
S=json.loads((R/'story.json').read_text());O=R/'output';O.mkdir(exist_ok=True)
W,H,FPS=720,1280,12
BG='#101923';PANEL='#1d2b38';WHITE='#f4f6f7';MUT='#b1bfcb';CYAN='#68e4cd';AMBER='#f2c77c';RED='#f28d8d';LINE='#435969'
FONTS={}
def font(size,b=False):
    key=(size,b)
    if key not in FONTS:FONTS[key]=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial'+(' Bold' if b else '')+'.ttf',size)
    return FONTS[key]
def text(d,s,x,y,size=27,col=WHITE,b=False,width=610,bottom=1240):
    for paragraph in s.split('\n'):
        line='';lines=[]
        for word in paragraph.split():
            trial=(line+' '+word).strip()
            if d.textlength(trial,font=font(size,b))>width and line:lines.append(line);line=word
            else:line=trial
        lines.append(line)
        for line in lines:
            assert d.textlength(line,font=font(size,b))<=width,(line,width)
            assert y+size*1.25<=bottom,(line,y,bottom)
            d.text((x,y),line,font=font(size,b),fill=col);y+=int(size*1.26)
    return y
def panel(d,xy,col=PANEL,outline=LINE):d.rounded_rectangle(xy,18,fill=col,outline=outline,width=2)
def arrow(d,a,b,col=CYAN):
    d.line([a,b],fill=col,width=4);theta=math.atan2(b[1]-a[1],b[0]-a[0]);d.polygon([b,(b[0]-14*math.cos(theta-.45),b[1]-14*math.sin(theta-.45)),(b[0]-14*math.cos(theta+.45),b[1]-14*math.sin(theta+.45))],fill=col)
def smooth(v):v=max(0,min(1,v));return v*v*(3-2*v)
def analyst(d,t):
    x,y=112,850
    d.line((x-15,y-70,x-20,y),fill='#486d83',width=20);d.line((x+15,y-70,x+25,y),fill='#486d83',width=20)
    d.line((x-34,y,x-7,y),fill=MUT,width=10);d.line((x+13,y,x+41,y),fill=MUT,width=10)
    d.polygon([(x-31,y-194),(x+31,y-194),(x+38,y-70),(x-38,y-70)],fill='#507b92')
    d.rectangle((x-9,y-214,x+9,y-186),fill='#dca57d');d.ellipse((x-31,y-271,x+31,y-205),fill='#dca57d')
    d.pieslice((x-33,y-277,x+32,y-222),180,360,fill='#283443')
    d.ellipse((x-13,y-242,x-8,y-237),fill=BG);d.ellipse((x+9,y-242,x+14,y-237),fill=BG)
    d.arc((x-8,y-228,x+13,y-214),0,155,fill=BG,width=2)
    hx=x+78+14*smooth(t/1.5);hy=y-158
    d.line([(x+25,y-178),(x+53,y-150),(hx,hy)],fill='#507b92',width=15);d.ellipse((hx-8,hy-8,hx+8,hy+8),fill='#dca57d')
    d.line([(x-25,y-179),(x-48,y-139),(x-20,y-123)],fill='#507b92',width=15)
    text(d,'ALEX',x-22,y-145,14,b=True,width=55,bottom=y-110)
def run(cmd):
    p=subprocess.run(cmd,capture_output=True,text=True,timeout=180)
    if p.returncode:raise RuntimeError(p.stderr[-1500:])
    return p.stdout
def duration(path):return float(run(['ffprobe','-v','error','-show_entries','format=duration','-of','csv=p=0',str(path)]))
def make_cues(events):
    groups=[];part=[]
    for e in events:
        part.append(e)
        if len(' '.join(x['text'] for x in part))>58:groups.append(part);part=[]
    if part:groups.append(part)
    return [(g[0]['offset']/1e7,(g[-1]['offset']+g[-1]['duration'])/1e7,' '.join(html.unescape(x['text']) for x in g)) for g in groups]
TIMELINE=[];TOTAL=0
for i,scene in enumerate(S['scenes']):
    audio=R/'assets'/f'narration-{i+1}.mp3';record=json.loads(audio.with_suffix('.json').read_text())
    assert record['text']==scene['voice'] and record['voice']==S['voice']
    length=math.ceil((duration(audio)+.2)*FPS)/FPS
    TIMELINE.append({'scene':scene,'id':i,'start':TOTAL,'duration':length,'cues':make_cues(record['events'])});TOTAL+=length
def draw(si,t,captions=True):
    seg=TIMELINE[si];u=t/seg['duration'];im=Image.new('RGB',(W,H),BG);d=ImageDraw.Draw(im)
    d.rectangle((0,0,7,H),fill=CYAN);text(d,'GERO  /  COMPUTATIONAL STUDY',45,67,20,CYAN,True,bottom=102)
    text(d,seg['scene']['title'],45,132,42,b=True,width=610,bottom=257)
    text(d,'KNOWN METHODS  ·  LOCAL PYTHON TEST',45,265,18,MUT,bottom=302)
    if si==0:
        panel(d,(47,330,673,490));pts=[(70+31*j,414-55*v) for j,v in enumerate([0,.25,-.2,.5,.1,-.6,-.3,.35,.55,-.1,.2,.7,.3,-.4,-.2,.35,.1,.45,.2])]
        d.line(pts,fill=CYAN,width=3)
        for j,p in enumerate(pts):d.ellipse((p[0]-4,p[1]-4,p[0]+4,p[1]+4),fill=WHITE if j/19<u else MUT)
        text(d,'Same observed nodes (schematic)',76,449,22,MUT,width=564,bottom=483)
        analyst(d,t)
        for j,name in enumerate(['Local formula','Telescoped formula']):
            y=545+150*j;panel(d,(245,y,669,y+113));text(d,name,267,y+20,27,b=True,width=382,bottom=y+67);text(d,'same estimate J',267,y+64,24,CYAN,width=382,bottom=y+108)
        arrow(d,(392,490),(392,542));arrow(d,(591,490),(660,690))
        text(d,'Can equal answers take less work?',53,887,27,AMBER,True,width=610,bottom=932)
    elif si==1:
        text(d,'Cubic endpoint differences telescope',52,330,28,CYAN,True,width=606,bottom=408)
        for j,name in enumerate(['w1 cubed','w2 cubed','w3 cubed']):
            y=420+109*j;fade=smooth((u-.10-j*.12)/.30);col=LINE if fade>.65 else WHITE
            panel(d,(60,y,314,y+79));panel(d,(407,y,661,y+79));text(d,'+ '+name,81,y+22,26,col,width=223,bottom=y+77);text(d,'- '+name,428,y+22,26,col,width=223,bottom=y+77)
            arrow(d,(322,y+40),(398,y+40),AMBER)
            if fade>.65:d.line((72,y+38,303,y+38),fill=RED,width=3);d.line((419,y+38,649,y+38),fill=RED,width=3)
        panel(d,(58,785,662,913),outline=CYAN)
        text(d,'One final cube - trapezoid sum',83,807,30,CYAN,True,width=560,bottom=902)
    elif si==2:
        text(d,'64 paths x 3 evaluations; 64 intervals',53,327,24,MUT,width=600,bottom=393)
        for j,(name,value,col) in enumerate([('Local loop',1.904,AMBER),('Telescoped',.520,CYAN)]):
            y=419+153*j;text(d,name,57,y,30,b=True,width=330,bottom=y+50);text(d,f'{value:.3f} ms',439,y,29,col,True,width=225,bottom=y+50)
            d.rounded_rectangle((59,y+54,59+int(576*value/1.904*smooth(t/1.7)),y+95),9,fill=col)
        panel(d,(55,738,667,915),outline=CYAN);text(d,'SAME MATHEMATICAL ERROR',78,762,25,CYAN,True,width=562,bottom=811)
        text(d,'1.45x with full test-data costs',78,819,28,WHITE,True,width=562,bottom=896)
    elif si==3:
        analyst(d,t);panel(d,(240,370,665,474),outline=CYAN);text(d,'COMMON MOMENTS',264,402,29,CYAN,True,width=379,bottom=466)
        for j in range(3):
            x=252+j*144;arrow(d,(457,479),(x+47,548))
            panel(d,(x,554,x+113,643));text(d,'f'+str(j+1),x+32,577,32,b=True,width=75,bottom=635)
        panel(d,(238,695,672,913),outline=AMBER);text(d,'32 queries: reuse helps',256,718,27,CYAN,True,width=389,bottom=790);text(d,'1 query: setup loses',256,781,27,AMBER,width=389,bottom=850);text(d,'Rounding can get worse',256,847,24,RED,True,width=391,bottom=906)
    else:
        analyst(d,t);panel(d,(237,351,672,893),outline=CYAN)
        for j,label in enumerate(['Exact identities','Executable code','Raw measurements','Negative controls']):
            y=389+j*88;d.rectangle((257,y+6,280,y+29),outline=CYAN,width=2);text(d,label,295,y,26,b=True,width=349,bottom=y+76)
        text(d,'No new integral.\nNo BSM correction.',259,779,28,AMBER,True,width=371,bottom=887)
        text(d,'Read the evidence and its limits.',53,912,26,CYAN,True,width=610,bottom=950)
    d.line((45,954,672,954),fill=LINE,width=2)
    if captions:
        cue=next((s for a,b,s in seg['cues'] if a<=t<b),None)
        if cue:text(d,cue,47,982,29,WHITE,width=604,bottom=1129)
    text(d,'Implementation-specific results · Not peer reviewed',45,1150,18,MUT,width=615,bottom=1191)
    text(d,'GERO.UZ  /  CODE + FULL LIMITATIONS',45,1198,19,CYAN,True,width=610,bottom=1240)
    for j in range(5):d.rectangle((45+126*j,1260,153+126*j,1264),fill=CYAN if j<=si else LINE)
    return im
def stamp(t):
    ms=round(t*1000);h,ms=divmod(ms,3600000);m,ms=divmod(ms,60000);s,ms=divmod(ms,1000);return f'{h:02}:{m:02}:{s:02},{ms:03}'
def main():
    p=argparse.ArgumentParser();p.add_argument('--proof',action='store_true');args=p.parse_args();assert 25<=TOTAL<=60,TOTAL
    (R/'measured-storyboard.json').write_text(json.dumps({'total':TOTAL,'scenes':TIMELINE},indent=2)+'\n')
    sheet=Image.new('RGB',(1080,1280),BG)
    for j,(si,fraction) in enumerate([(0,.2),(1,.2),(1,.85),(2,.8),(3,.65),(4,.6)]):
        im=draw(si,TIMELINE[si]['duration']*fraction);im.save(O/f'proof-{j+1}.png');sheet.paste(im.resize((360,640)),((j%3)*360,(j//3)*640))
    sheet.save(O/'contact-sheet.jpg');draw(2,TIMELINE[2]['duration']*.9,False).save(O/'cover.png')
    if args.proof:print('PROOF',TOTAL);return
    parts=[];cues=[]
    for seg in TIMELINE:
        si=seg['id'];frames=O/'frames'/str(si);frames.mkdir(parents=True,exist_ok=True)
        for n in range(round(seg['duration']*FPS)):draw(si,n/FPS).save(frames/f'{n:05}.png')
        target=O/f'part-{si}.mp4'
        run(['ffmpeg','-y','-v','error','-threads','1','-filter_threads','1','-filter_complex_threads','1','-framerate',str(FPS),'-threads','1','-i',str(frames/'%05d.png'),'-threads','1','-i',str(R/'assets'/f'narration-{si+1}.mp3'),'-t',str(seg['duration']),'-c:v','libx264','-threads','1','-preset','ultrafast','-crf','22','-pix_fmt','yuv420p','-c:a','aac','-b:a','128k','-ar','48000','-af','apad',str(target)])
        parts.append(target);cues.extend((a+seg['start'],b+seg['start'],s) for a,b,s in seg['cues']);print('encoded',si+1,flush=True)
    listing=O/'concat.txt';listing.write_text(''.join("file '"+str(x)+"'\n" for x in parts));target=O/'ito-computation.mp4'
    run(['ffmpeg','-y','-v','error','-threads','1','-f','concat','-safe','0','-i',str(listing),'-c','copy','-movflags','+faststart',str(target)])
    length=duration(target);assert abs(length-TOTAL)<.15;assert all(0<=a<b<=length+.04 for a,b,s in cues)
    (O/'captions.en.srt').write_text('\n\n'.join(str(i)+'\n'+stamp(a)+' --> '+stamp(b)+'\n'+s for i,(a,b,s) in enumerate(cues,1))+'\n')
    run(['ffmpeg','-v','error','-threads','1','-filter_threads','1','-filter_complex_threads','1','-i',str(target),'-map','0','-f','null','-'])
    receipt={'status':'rendered_full_decode_pass_visual_QA_pending','duration_seconds':length,'width':W,'height':H,'fps':FPS,'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'bytes':target.stat().st_size,'caption_cues':len(cues),'caption_source':'actual edge-tts WordBoundary events','cpu_threads':1,'gpu':False,'playback':False,'human_audible_QA':False,'voice':S['voice'],'rate':S['rate'],'published':False}
    (O/'verification.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
if __name__=='__main__':main()

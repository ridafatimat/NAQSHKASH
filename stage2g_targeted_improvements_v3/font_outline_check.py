"""Read-only TrueType simple-outline comparison; no glyph rasterization."""
import struct,json,hashlib,csv
from pathlib import Path
FONT=Path('C:/FYP Dataset Generation/Assets/Talamat.ttf')
OUT=Path('C:/NAQSHKASH/stage2/stage2g_targeted_improvements_v3')
b=FONT.read_bytes()
def u16(p):return struct.unpack_from('>H',b,p)[0]
def s16(p):return struct.unpack_from('>h',b,p)[0]
def u32(p):return struct.unpack_from('>I',b,p)[0]
tables={}
for i in range(u16(4)):
 p=12+16*i;tag=b[p:p+4].decode('ascii');tables[tag]=(u32(p+8),u32(p+12))
head=tables['head'][0];num=u16(tables['maxp'][0]+4);loc=tables['loca'][0]
offsets=[u16(loc+2*i)*2 for i in range(num+1)] if s16(head+50)==0 else [u32(loc+4*i) for i in range(num+1)]
cmap=tables['cmap'][0];choices=[]
for i in range(u16(cmap+2)):
 p=cmap+4+8*i;start=cmap+u32(p+4)
 if u16(start)==4:choices.append((u16(p)==3 and u16(p+2)==1,start))
assert choices,'No supported format4 cmap; do not infer equivalence'
start=max(choices)[1];segments=u16(start+6)//2;ends=start+14;starts=ends+2*segments+2;deltas=starts+2*segments;ranges=deltas+2*segments
def glyph_index(ch):
 code=ord(ch)
 for i in range(segments):
  if u16(starts+2*i)<=code<=u16(ends+2*i):
   d=s16(deltas+2*i);r=u16(ranges+2*i)
   if not r:return (code+d)&65535
   value=u16(ranges+2*i+r+2*(code-u16(starts+2*i)))
   return ((value+d)&65535) if value else 0
 return 0
def outline(ch):
 gid=glyph_index(ch);assert 0<gid<num,(ch,gid)
 p=tables['glyf'][0]+offsets[gid];n=s16(p)
 if n<0:return dict(status='composite_not_compared',glyph_id=gid)
 if n==0:return dict(status='empty_not_compared',glyph_id=gid)
 declared=tuple(s16(p+2+2*i) for i in range(4));endpts=[u16(p+10+2*i) for i in range(n)];count=endpts[-1]+1;p+=10+2*n;instr=u16(p);p+=2+instr;flags=[]
 while len(flags)<count:
  flag=b[p];p+=1;repeat=b[p] if flag&8 else 0;p+=bool(flag&8);flags.extend([flag]*(repeat+1))
 assert len(flags)==count
 axes=[]
 for short,same in [(2,16),(4,32)]:
  values=[];v=0
  for f in flags:
   if f&short:delta=b[p]*(1 if f&same else -1);p+=1
   elif f&same:delta=0
   else:delta=s16(p);p+=2
   v+=delta;values.append(v)
  axes.append(values)
 xs,ys=axes;minx,miny=min(xs),min(ys);pts=[(x-minx,y-miny,int(f&1)) for x,y,f in zip(xs,ys,flags)];contours=[];begin=0
 for end in endpts:
  ring=pts[begin:end+1];begin=end+1
  # Contour start point and traversal direction do not change the outline.
  variants=[tuple(z[i:]+z[:i]) for z in [ring,ring[::-1]] for i in range(len(ring))];contours.append(min(variants))
 canonical=tuple(sorted(contours))
 assert tuple([min(xs),min(ys),max(xs),max(ys)])==declared,'Outline bounds must agree with font glyph header'
 return dict(status='simple_outline',glyph_id=gid,points=count,contours=n,coordinate_bounds=[min(xs),min(ys),max(xs),max(ys)],declared_bounds=declared,translation_invariant_outline_sha256=hashlib.sha256(repr(canonical).encode()).hexdigest(),ordered_translation_sha256=hashlib.sha256(repr((pts,endpts)).encode()).hexdigest(),canonical=canonical)
rows=[];details={}
for a,c in [('E','Z'),('C','Y'),('K','^'),('G','['),('I','\\')]:
 aa=outline(a);cc=outline(c);comparable=aa['status']==cc['status']=='simple_outline';equal=comparable and aa['canonical']==cc['canonical'];details[a]={k:v for k,v in aa.items() if k!='canonical'};details[c]={k:v for k,v in cc.items() if k!='canonical'}
 rows.append(dict(upper=a,lower=c,upper_glyph=aa['glyph_id'],lower_glyph=cc['glyph_id'],simple_outline_comparison=comparable,translation_equivalent_outlines=equal,ordered_translation_equivalent=comparable and aa['ordered_translation_sha256']==cc['ordered_translation_sha256'],upper_points=aa.get('points',''),lower_points=cc.get('points',''),upper_bounds=aa.get('coordinate_bounds',''),lower_bounds=cc.get('coordinate_bounds','')))
with (OUT/'cross_role_font_outline_evidence.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(OUT/'source_font_evidence.json').write_text(json.dumps(dict(font=str(FONT),sha256=hashlib.sha256(b).hexdigest(),method='Read existing TrueType simple contours via cmap/loca/glyf; compare translated canonical points, contour boundaries and on-curve flags. No rasterization, generated image, training or inference. Exact equivalence is stronger than visual similarity; a negative result does not prove dissimilarity.',glyphs=details),indent=2))
print(json.dumps(rows,indent=2))

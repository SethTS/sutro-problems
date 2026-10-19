
import triton,triton.language as tl
from triton.language import constexpr as G,float16 as A3,float32 as A1,float64 as Ad,int32 as C,int64 as Ae,program_id as X,arange as I,load as H,sum as A0,full as A7,store as Y,sqrt as D,dot as A2,cdiv as L,where as A4,constexpr as A,float32 as K,float16 as E,maximum as A8,trans as A6
from torch import empty as U,cat as i,eye as o,mm as s,empty_like as c,int32 as S,ones as p,float16 as d,float32 as k,zeros_like as q,linalg as f,cuda as j,bool as r
@triton.jit
def R(q,r,U,e,t:G,w:G,u:G,s:G,v:G):
	C=X(0);R=I(0,128);O=I(0,s);P=H(U+C*u+R,R<u,0);g=H(r+P[:,None]*t+O[None,:],(R[:,None]<u)&(O[None,:]<t),0);T=A0(g,0)/u;m=A4(R[:,None]<u,g-T[None,:],.0).to(A3);x=A2(A6(m),m)+A4(O[:,None]==O[None,:],v,.0);B=H(q+C//w*t+O,O<t,0)-T;o=m.to(A1);k=A0(A0(o*o,0),0);L=v-2*(7.62939453125e-06/.9999923706054688)*(k+v);p=A7((s,),.0,A1);h,f=B,B;a=A0(h*h,0);Q=a;j=0;c=1.;d=.0;J=1.;E=L;l=.0;S=.0;N=float('inf');M=L>0
	while(j<32)&(a>Q*1e-12)&((j<4)|(N-S>.01*A8(S,1e-30))|(N<S)|(N!=N)|(S!=S)|~M):
		A=A0(x*f[None,:],1);F=A0(f*A,0);z=a/A8(F,1e-30)
		if j>0:K=1.-c*E;M&=K>0;E=L+d*E/A8(K,1e-30);J=1.+d*J
		N=S+a/E*(2*l+J/E);S+=a*z*(2*l+z*J);l+=z*J;p+=z*f;h-=z*A;b=A0(h*h,0);D=b/A8(a,1e-30);f=h+f*D;a=b;c=z;d=D;j+=1
	n=v*v*A0(p*p,0);i=v*v*(S+N)*.5;y=M&(N>=S)&(N-S<=.01*A8(S,1e-30));Y(e+C,A4(y,i,n))
@triton.jit
def B(d,h,b:G,a:G,c:G):
	f=I(0,a);g=(f[:,None]<b)&(f[None,:]<b);o=H(d+f[:,None]*b+f[None,:],g,.0);e=((f[:,None]==f[None,:])&g).to(A1);j=D(A0(A0(o*o,0),0));p,q=o/j,e
	for l in range(c):m=1.5*e-.5*A2(q,p,input_precision='tf32x3');p,q=A2(p,m,input_precision='tf32x3'),A2(q,m,input_precision='tf32x3')
	i=p*D(j);k=D(A0(A0(i*i,0),0));p,q=i/k,e
	for l in range(c):m=1.5*e-.5*A2(q,p,input_precision='tf32x3');p,q=A2(p,m,input_precision='tf32x3'),A2(q,m,input_precision='tf32x3')
	o=A2(i,p*D(k),input_precision='tf32x3');o=(o+A6(o))*.5;n=A0(A0(o*e,0),0);o=.98*o*(b/A8(n,1e-12))+.02*e;Y(h+f[:,None]*b+f[None,:],o,g)
M=4096
@triton.jit
def Z(f,l,i,d,b:G,a:G):j=X(0)*a+I(0,a);c=I(0,16);m=H(f+j,j<b,0);g=H(l+j,j<b,0)&(j<b);e=(c[:,None]==m[None,:])&g[None,:];k=tl.cumsum(e.to(C),1);h=A0(A4(e,k,0),0)-1;Y(i+j,h,j<b);Y(d+X(0)*16+c,A0(e.to(C),1))
@triton.jit
def a(k,s,q,m,o,j,c:G,b:G,a:G,d:G,e:G):
	g=X(0);h=I(0,e);i=I(0,16);n=H(m+h[:,None]*16+i[None,:],h[:,None]<d,0);f=A0(A4(h[:,None]<g,n,0),0);r=g*a+I(0,a);t=H(k+r,r<c,0).to(C);l=H(s+r,r<c,0)&(r<c);p=H(q+r,r<c,0)+tl.gather(f,t,0);Y(o+t*b+p,r,l&(p<b))
	if g==0:Y(j+i,A0(n,0),i<10)
def t(labels,valid,cap):E=valid;A=labels;B=len(A);D=512;C=triton.cdiv(B,D);F=U(B,device=A.device,dtype=S);G=U(C,16,device=A.device,dtype=S);H=torch.full((10,cap),B,device=A.device,dtype=torch.long);I=U(10,device=A.device,dtype=S);Z[C,](A,E,F,G,B,D,num_warps=4);a[C,](A,E,F,G,H,I,B,cap,D,C,triton.next_power_of_2(C),num_warps=4);return H,I
@triton.jit
def b(t,u,p,i,m,l,d:G,e:G,f:G,c:G,g:G,b:G,a:G):
	h=X(0)
	if(X(1)*b<H(m+h))&(X(2)*a<H(i+h)):n=X(1)*b+I(0,b);q=X(2)*a+I(0,a);j=I(0,64);o=H(p+h*g+n,n<g,f);r=H(t+o[:,None]//e*d+j[None,:],(o[:,None]<f)&(j[None,:]<d),.0);s=H(u+(h*c+q[None,:])*d+j[:,None],(q[None,:]<c)&(j[:,None]<d),.0);k=A8(A0(r*r,1)[:,None]+A0(s*s,0)[None,:]-2*A2(r,s,input_precision='tf32x3'),.0);k=A4(q[None,:]<H(i+h),k,float('inf'));Y(l+o[:,None]*c+q[None,:],k,(o[:,None]<f)&(q[None,:]<c))
def AC(zq,zv,yv,valid,cls):B=cls;A,C=B.shape;D,F=t(yv,valid,M);G,H=t(B.flatten(),p(B.numel(),device=B.device,dtype=r),A);I=i([zv,q(zv[:1])])[D].contiguous();E=torch.full((A*C,M),float('inf'),device=zq.device,dtype=d);b[10,triton.cdiv(A,32),triton.cdiv(M,128)](zq,I,G,F.clamp_max(M),H,E,zq.shape[1],C,A*C,M,A,32,128,num_warps=4,enable_fp_fusion=False);L,J=AU(E,O);K=D[B.flatten()[:,None],J].reshape(A,C,O);return K
def AD(zq,zv,yv,valid,cls):B,A=cls.shape;E=AC(zq,zv,yv,valid,cls);C=zq.shape[1];D=U(B*A,device=zq.device);R[B*A,](zq,zv,E,D,C,A,O,triton.next_power_of_2(C),AI,num_warps=4,enable_fp_fusion=False);return D.reshape(B,A)
AE=49
@triton.jit
def e(k,l,i,c:G,b:G,a:G):j=X(0)*a+I(0,a);f=I(0,16);h=j[:,None]*b+f[None,:];g=(j[:,None]<c)&(f[None,:]<b);d=H(k+h,g,0);e=H(l+h,g,0);Y(i+X(0)*16+f,A0(d*e,0))
@triton.jit
def V(r,h,s,n,m,k,j,c:G,b:G,a:G,d:G,e:G):
	u,g=I(0,e),I(0,16);o=A0(H(s+u[:,None]*16+g[None,:],u[:,None]<d,0),0);q=H(n+g,g<b,1);f=o/q;v=X(0)*a+I(0,a);p=v[:,None]*b+g[None,:];l=(v[:,None]<c)&(g[None,:]<b);t,i=H(r+p,l,0),H(h+p,l,0);w=t+i*f[None,:];Y(k+p,w,l);Y(j+p,w,l)
	if X(0)==0:Y(m+g,o,g<b)
@triton.jit
def J(l,j,c:G,b:G,d:G,a:G):
	h=X(0)*a+I(0,a);k,f=h//b,h%b
	if d:e=H(l+k*(2*b)+f,k<c,other=0);e+=H(l+k*(2*b)+f+b,k<c,other=0)
	else:e=H(l+h,k<c,other=0)
	g=e.to(A3);i=(e-g.to(A1)).to(A3);Y(j+k*(2*b)+f,g,k<c);Y(j+k*(2*b)+f+b,i,k<c)
import gc,math,torch,torch.nn.functional as F
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False
N,w,Af,Ag,AF=.02,.1,256,.02,.6
Ah=7
AG,AH=.2,512
O,AI,AJ,AK,AL,Ai=70,8.,2.,.15,5,2048
Aj,Ak,Al,u=5000,1000,7,3.
P={}
def x(a):return f.cholesky_ex(a)[0]
def y(x):return x*(x.shape[1]**.5/x.norm(dim=1,keepdim=True).clamp_min(1e-06))**AF
def A9(a,b,shift=.0):A=a.square().sum(1,keepdim=True);B=b.square().sum(1,keepdim=True);C=i([a,A,torch.ones_like(A)],1);D=i([2*N*b,torch.full_like(B,-N),shift-N*B],1).T.contiguous();return C,D
def AM(a,b):A,B=A9(a,b);return(A@B).exp_()
def AN(z):C,D=A9(z,z,u);A,B=C.half(),D.half();H,I=(C-A.float()).half(),(D-B.float()).half();E,J=i([A,A,H],1),i([B,I,B],0);G=-E.shape[1]%16;return F.pad(E,(0,G)),F.pad(J,(0,0,0,G))
def AO(d,t):A=d.min(1,keepdim=True).values;return torch.softmax(-(d-A)/(t*A).clamp_min(1e-12),1)
def AP(tx,y):D,E=tx.shape;G=y.sum(0);A=(y.T@tx*(G*D).rsqrt()[:,None]).T.double();C=1.-1./D+.001;F=o(E,device=tx.device,dtype=torch.float64);H=F[:10,:10]-A.T@A/C;I=x(H);J=torch.cholesky_solve(A.T,I);B=F/C+3./C**2*(A@J);B=(B*(E/B.trace())).float();K=x(B);return K
def z(train_x,train_y,test_x):
	N=train_y;O=train_x;P=O;H=N;B=math.exp(u);X,Y,d=w*B,.0001*B,B*(1.+w)
	with torch.no_grad():E,Z=y(P.float()),y(test_x.float());G=len(E);I=F.one_hot(H.long(),10).float();J=AP(P,I);e=j.current_stream();f=j.Stream();J=AX(O,E,I,N,J);A=i([E,Z])@J;Q=A[:G].square().sum(1).mean();R=A.norm(dim=1,keepdim=True).clamp_min(1e-06);K=Q.sqrt();A=i([A*(K/R),.5*K*(R/K).log()],1);A=A*(Q/A[:G].square().sum(1).mean()).sqrt();L=Ac(A,H,I,B,X,Y);S=L.topk(2,1).values;T=S[:,0]-S[:,1];U=T>=AG;M=L.argmax(1);C=T.topk(AH,largest=False).indices;D=L[C]/B;V=D.topk(AL,1).indices;a=i([H.long(),M]);b=i([p(G,dtype=r,device=E.device),U]);c=AO(AD(A[G:][C],A,a,b,V),AJ);D=D+AK*q(D).scatter_(1,V,c);W=M.clone();W[C]=torch.where(U[C],M[C],D.argmax(1));return W
def rfm_d3(train_x,train_y,test_x):
	D=test_x;E=train_y;B=train_x;F=tuple(B.shape),tuple(D.shape),B.dtype,E.dtype,str(B.device)
	if F not in P:
		A=[B.clone(),E.clone(),D.clone()];G=j.Stream();G.wait_stream(j.current_stream())
		with j.stream(G):
			for I in range(2):z(*A)
		j.current_stream().wait_stream(G);C=j.CUDAGraph()
		with j.graph(C):H=z(*A)
		P[F]=C,A,H;gc.collect();gc.freeze()
		for J in range(64):C.replay()
		j.synchronize()
	C,A,H=P[F];A[0].copy_(B);A[1].copy_(E);A[2].copy_(D);C.replay();return H.clone()
AA,Q=128,4
@triton.jit
def g(D,q,j,A:G,z:G,y:G,v:G,u:G,t:G,s:G,C:G,x:G,w:G,B:G):
	l=X(0)*t+I(0,t);g=I(0,64);f=I(0,s);c=I(0,16);h:G=2*(v+2);F=H(D+(z+l[:,None])*y+g[None],(l[:,None]<A)&(g[None]<v),0);n=H(D+(z+l)*y+v,l<A,0).to(A1)+H(D+(z+l)*y+h+v,l<A,0).to(A1);E=A7((t,16),0,A1);b=L(z,s*C)
	for o in range(b):e=(X(1)*b+o)*s+f;a=H(D+e[None]*y+g[:,None],(g[:,None]<v)&(e[None]<z),0);d=A2(F,a);m=H(D+e*y+v,e<z,0).to(A1)+H(D+e*y+h+v,e<z,0).to(A1);i=(B+2*w*d-w*(n[:,None]+m[None])).to(A3).to(A1);r=tl.exp(i).to(A3);p=H(q+e[:,None]*u+c[None],(e[:,None]<z)&(c[None]<u),0);k=A2(r,p);E=tl.inline_asm_elementwise('add.rn.f32 $0, $1, $2;',constraints='=f,f,f',args=[E,k],dtype=A1,is_pure=True,pack=1)
	Y(j+(X(1)*A+l[:,None])*u+c[None],E,(l[:,None]<A)&(c[None]<u))
def AQ(a,b,n,w):B=w.abs().amax(0,keepdim=True).clamp_min(1e-30);D=(w/B).half();A=len(a)-n;C=U((Q,A,w.shape[1]),device=w.device);g[triton.cdiv(A,AA),Q](a,D,C,A,n,a.shape[1],AE,w.shape[1],AA,64,Q,False,N,u,num_warps=4);E=C.sum(0)*B;return E
def v(phi,inv,r,rh=None):
	C=inv[0];B,A=r.shape
	if rh is None:rh=U((B,2*A),device=r.device,dtype=d);J[triton.cdiv(B*A,1024),](r,rh,B,A,False,1024,num_warps=4)
	E=s(phi.T,rh,out_dtype=k);H=C@(E[:,:A].double()+E[:,A:].double());F=U((len(C),2*A),device=r.device,dtype=d);J[triton.cdiv(len(C)*A,1024),](H,F,len(C),A,False,1024,num_warps=4);I=s(phi,F,out_dtype=k);G=c(r);D=U((triton.cdiv(B,256),16),device=r.device);m[len(D),](I,r,G,D,inv[1],B,A,256,num_warps=4,enable_fp_fusion=False);return G,D
def AR(op,t,steps,phi,inv):
	Q=steps;K=inv;L=phi;E,F=t.shape;B=triton.cdiv(E,256);M=U((B,16),device=t.device);N,I=q(t),t.clone();A,T=v(L,K,I);C,G=T.sum(0),A.half();O=U((E,2*F),device=t.device,dtype=d)
	for R in range(Q):
		S=op.mv(G,R%2==1);e[B,](A,S,M,E,F,256,num_warps=4);h[B,](N,I,A,S,M,C,O,E,F,256,B,triton.next_power_of_2(B),num_warps=4,enable_fp_fusion=False)
		if R+1==Q:break
		P,D=v(L,K,I,O);H,G=c(A),c(A,dtype=d);J=c(C);V[B,](P,A,D,C,J,H,G,E,F,256,len(D),triton.next_power_of_2(len(D)),num_warps=4,enable_fp_fusion=False);A,C=H,J
	P,D=v(L,K,I,O);H,G=c(A),c(A,dtype=d);J=c(C);V[B,](P,A,D,C,J,H,G,E,F,256,len(D),triton.next_power_of_2(len(D)),num_warps=4,enable_fp_fusion=False);return N+H*(C/M.sum(0));return N
@triton.jit
def h(x,t,j,h,r,p,u,c:G,b:G,a:G,d:G,e:G):s,i=I(0,e),I(0,16);l=A0(H(r+s[:,None]*16+i[None],s[:,None]<d,0),0);g=H(p+i,i<b,0)/l;v=X(0)*a+I(0,a);q=v[:,None]*b+i[None];o=(v[:,None]<c)&(i[None]<b);y,w=H(x+q,o,0),H(t+q,o,0);k,f=H(j+q,o,0),H(h+q,o,0);w=w-f*g[None];Y(x+q,y+k*g[None],o);Y(t+q,w,o);m=w.to(A3);n=(w-m.to(A1)).to(A3);Y(u+v[:,None]*(2*b)+i[None],m,o);Y(u+v[:,None]*(2*b)+i[None]+b,n,o)
def AS(a):A=c(a);B[1,](a,A,len(a),triton.next_power_of_2(len(a)),7,num_warps=4,enable_fp_fusion=False);return A
@triton.jit
def T(f,a,e,b:G,c:G,d:G,g:G):i=X(0);j,m=i//c,i%c;h=I(0,g);l=I(0,64);q=H(f+((j*d+h[:,None])*c+m)*64+l[None,:],h[:,None]<d,0);p=A0(q,0);k=A0(A4(l==b,p,0),0);o=H(a+i*b+l,l<b,0);n=A4(k>0,p/A8(k,1),o);Y(e+i*b+l,n,l<b)
def AT(z,labels,mc=32):
	A=mc;D=1400;F,J=t(labels,p(len(z),device=z.device,dtype=r),D);G=i([z,q(z[:1])])[F].contiguous();I=F<len(z);B=G[:,:A].reshape(10*A,-1).contiguous();C=triton.cdiv(D,128);E=U((10,C,A,64),device=z.device,dtype=k)
	for K in range(2):H=c(B);W[10,C](G,I,B,E,E,D,z.shape[1],A,triton.next_power_of_2(A),C,128,False,True,num_warps=4,enable_fp_fusion=False);T[10*A,](E,B,H,z.shape[1],A,C,triton.next_power_of_2(C),num_warps=4,enable_fp_fusion=False);B=H
	return B
@triton.jit
def l(i,s,k,c:G,b:G,a:G):
	p=X(0);g=I(0,a);r=H(i+p*c+g,g<c,0);f=r.to(tl.uint16,bitcast=True).to(C)&32767;m=A7((),0,C)
	for q in range(15):e=m|1<<14-q;h=A0(((f<e)&(g<c)).to(C),0);m=A4(h<b,e,m)
	d=(f<m)&(g<c);j=(f==m)&(g<c);n=b-A0(d.to(C),0);l=d|j&(tl.cumsum(j.to(C),0)<=n);o=tl.cumsum(l.to(C),0)-1;Y(s+p*b+o,r,l);Y(k+p*b+o,g,l)
def AU(distance,k=70):A=distance;B,C=A.shape;D=U((B,k),device=A.device,dtype=A.dtype);E=U((B,k),device=A.device,dtype=torch.int64);l[B,](A,D,E,C,k,triton.next_power_of_2(C),num_warps=8);return D,E
@triton.jit
def m(l,m,k,h,g,c:G,b:G,a:G):n=X(0)*a+I(0,a);f=I(0,16);i=(n[:,None]<c)&(f[None,:]<b);j=H(m+n[:,None]*b+f[None,:],i,0);d=H(l+n[:,None]*(2*b)+f[None,:],i,0);e=H(l+n[:,None]*(2*b)+b+f[None,:],i,0);o=H(g+n,n<c,0);p=j*o[:,None]-(d+e);Y(k+n[:,None]*b+f[None,:],p,i);Y(h+X(0)*16+f,A0(j*p,0))
@triton.jit
def n(l,k,c:G,b:G,d:G,a:G):
	i=X(0)*a+I(0,a);m,h=i//b,i%b;o=m<c;j=A4(o,m,m-c);p=A7((a,),.0,A1)
	for n in range(d):e=H(l+(n*c+j)*b+h,m<2*c,0);f=H(l+((d+n)*c+j)*b+h,m<2*c,0);g=H(l+((2*d+n)*c+j)*b+h,m<2*c,0);p+=A4(o,e,f+g)
	Y(k+i,p,m<2*c)
@triton.jit
def A5(r,s,u,q,d:G,e:G,c:G,b:G,a:G,g:G,f:G):
	o=X(2);v=X(0)*b+I(0,b);p=I(0,a);m=I(0,16);y=A4(o==0,d,e);l=L(y,a*g);i=A7((b,16),.0,A1)
	for x in range(l):
		k=l-1-x if f else x;n=(X(1)*l+k)*a+p
		if o==1:h=H(r+n[None,:]*d+e+v[:,None],(v[:,None]<e)&(n[None,:]<y),0)
		else:t=A4(o==0,r,s);h=H(t+v[:,None]*y+n[None,:],(v[:,None]<e)&(n[None,:]<y),0)
		w=A4(o==2,e,0);j=H(u+(w+n[:,None])*c+m[None,:],(n[:,None]<y)&(m[None,:]<c),0);i=A2(h,j,i)
	Y(q+((o*g+X(1))*e+v[:,None])*c+m[None,:],i,(v[:,None]<e)&(m[None,:]<c))
class AV:
	def __init__(A,pans,pp,n):A.pans,A.pp,A.n=pans,pp,n
	def mv(A,dh,rev):B=dh;D=U((12,A.pp,B.shape[1]),device=B.device);C=U((A.n,B.shape[1]),device=B.device);A5[triton.cdiv(A.pp,128),4,3](A.pans[0],A.pans[1],B,D,A.n,A.pp,B.shape[1],128,128,4,rev,num_warps=4,enable_fp_fusion=False);n[triton.cdiv(C.numel(),512),](D,C,A.pp,B.shape[1],4,512,num_warps=4,enable_fp_fusion=False);return C
def AB(a,b):return torch.bmm(a.half(),b.half(),out_dtype=k)
def AW(m,rawx):A=m.double();E=rawx[:48].T.double();C=A@E;D=A@C;F=A@D;G=1e-07*A.square().sum()/48;H=E.T@C;I=C.T@D;J=D.T@F;B=J+G*I+G.square()*(H+o(48,device=m.device,dtype=torch.float64)*(1e-07*H.trace()/48));B=(B+B.T)*.5;K,M=f.cholesky_ex(B);L=f.solve_triangular(K,F.T,upper=False).T;return L.float()
def AX(raw,x,y,labels,initial):B=initial;N=f.cholesky_ex(.5*o(B.shape[0],device=x.device)+.5*(B@B.T))[0];F=B;Y,D=x.shape;E=2;O,A,G,P=AY(x,labels);Q=o(D,device=x.device);A=.6*A+.4*torch.cholesky_inverse(N);H,Z=f.cholesky_ex(A);R=torch.cholesky_inverse(H);I=(x[:2048][None]-O[:,None]).contiguous();J=-AB(I,R);S=torch.where(G>0,(G/P.float().repeat_interleave(E)).log(),float('-inf'));C=.5*(I*J).sum(2)-H.diagonal(dim1=1,dim2=2).log().sum(1)[:,None]+S[:,None];C=C.reshape(10,E,2048).permute(2,0,1);T=torch.softmax(C,2);U=C.logsumexp(2)/4.;K=torch.softmax(U,1);V=J.reshape(10,E,2048,D).permute(2,0,1,3);L=(V*T[:,:,:,None]).sum(2);W=L-(L*K[:,:,None]).sum(1,keepdim=True);M=(W*K.sqrt()[:,:,None]).reshape(-1,D).contiguous();A=AB(M.T[None].contiguous(),M[None])[0];X=.8*AS(A)+.196*(F@F.T)+.004*Q;return AW(X,x)
@triton.jit
def W(A,z,q,w,o,r:G,s:G,t:G,v:G,u:G,p:G,x:G,y:G):
	a,D=X(0),X(1);h=D*p+I(0,p);c=I(0,64);b=I(0,v);l=H(A+(a*r+h[:,None])*s+c[None,:],(h[:,None]<r)&(c[None,:]<s),0);E=H(q+(a*t+b[:,None])*s+c[None,:],(b[:,None]<t)&(c[None,:]<s),0)
	if x:m,F=l.to(A3),E.to(A3);n,G=(l-m.to(A1)).to(A3),(E-F.to(A1)).to(A3);i=A2(m,A6(F))+A2(n,A6(F))+A2(m,A6(G));i=i-.5*A0(E*E,1)[None,:]
	else:g=(-.5*A0(E*E,1)).to(A3).to(A1);i=(A2(l.to(A3),A6(E.to(A3)))+g[None,:]).to(A3).to(A1)
	i=A4(b[None,:]<t,i,float('-inf'));B=tl.argmax(i,1,tie_break_left=True)
	if x:Y(o+a*r+h,B,h<r)
	if y:k=H(z+a*r+h,h<r,0);f=((B[None,:]==b[:,None])&k[None,:]).to(A3);C=A4(c[None,:]==s,k[:,None].to(A1),l);d=C.to(A3);e=(C-d.to(A1)).to(A3);j=A2(f,d)+A2(f,e);Y(w+((a*u+D)*t+b[:,None])*64+c[None,:],j,b[:,None]<t)
@triton.jit
def mom2(j,i,c,g,e:G,f:G,d:G,h:G):
	s=X(0);p,q=s//2,s%2;r=I(0,64);u=I(0,d);k=A7((64,64),.0,A1);l=A7((64,64),.0,A1);n=A7((64,64),.0,A1);o=L(e,d*h)
	for A in range(o):z=(X(1)*o+A)*d+u;a=H(i+p*e+z,z<e,0);y=H(c+p*e+z,z<e,-1)==q;b=H(j+(p*e+z[:,None])*f+r[None,:],(z[:,None]<e)&(r[None,:]<f),0);m=A4(r[None,:]==f,1.,b);t=m.to(A3);v=(m-t.to(A1)).to(A3);w=A4((a&y)[:,None],t,.0);x=A4((a&y)[:,None],v,.0);k=A2(A6(w),t,k);l=A2(A6(w),v,l);n=A2(A6(x),t,n)
	Y(g+((s*h+X(1))*64+r[:,None])*64+r[None,:],k+n+l)
def AY(x,labels):
	I,A=x.shape;C=1400;J=64;K,Q=t(labels,p(I,device=x.device,dtype=r),C);F=i([x,q(x[:1])])[K].contiguous();L=K<I;E=F[:,:2].reshape(20,A).contiguous();D=triton.cdiv(C,J);M=U((10,D,2,64),device=x.device);N=U((10,C),device=x.device,dtype=S)
	for O in range(3):
		W[10,D](F,L,E,M,N,C,A,2,16,D,J,True,O<2,num_warps=4,enable_fp_fusion=False)
		if O<2:P=c(E);T[20,](M,E,P,A,2,D,triton.next_power_of_2(D),num_warps=4,enable_fp_fusion=False);E=P
	B=U((20,8,64,64),device=x.device);mom2[20,8](F,L,N,B,C,A,128,8,num_warps=4,enable_fp_fusion=False);B=B.sum(1);G=B[:,A,A];H=B[:,:A,A]/G.clamp_min(1)[:,None];R=B[:,:A,:A]/G.clamp_min(1)[:,None,None]-H[:,:,None]*H[:,None,:];return H,R,G,Q
@triton.jit
def radial(h,c,f,e:G,g:G,d:G,a:G,b:G):o=X(0)*a+I(0,a);k=X(1)*b+I(0,b);n=I(0,64);r=H(h+o[:,None]*d+n[None,:],(o[:,None]<e)&(n[None,:]<d),0);i=H(c+k[:,None]*d+n[None,:],(k[:,None]<g)&(n[None,:]<d),0);s=A0(r*r,1);j=A0(i*i,1);p=i*.04;m=A2(r,A6(p),input_precision='tf32x3');l=m-s[:,None]*.02-j[None,:]*.02;q=tl.exp(l);Y(f+o[:,None]*g+k[None,:],q,(o[:,None]<e)&(k[None,:]<g))
@triton.jit
def _construct(a,b,n,i,k:A,g:A,l:A,m:A,j:A,h:A,f:A,d:A,e:A,c:A):
	u=X(0)*d+I(0,d);r=X(1)*e+I(0,e);t=I(0,c);p=A7((d,e),.0,K)
	for v in range(L(l,c)):s=v*c+t;o=H(a+(j+u[:,None])*l+s[None],(u[:,None]<k)&(s[None]<l),0);q=H(b+s[:,None]*m+j+r[None],(s[:,None]<l)&(r[None]<g),0);p=A2(o,q,p)
	w=tl.exp(p.to(E).to(K)).to(E).to(K);y=H(n+j+u,u<k,0);x=H(n+j+r,r<g,0);w=w*D(y[:,None]*x[None,:]);w=A4(u[:,None]==r[None,:],f*y[:,None]+h,w).to(E);Y(i+u[:,None]*g+r[None],w,(u[:,None]<k)&(r[None]<g))
def AZ(_local__panels_a,_local__panels_b,_local__panels_n,_local__panels_mass,_local__panels_lam,_local__panels_c):
	E=_local__panels_b;D=_local__panels_a;B=_local__panels_n;A=B//2;F=[]
	for C in(0,A):G=U((A,B-C),device=D.device,dtype=d);_construct[triton.cdiv(A,128),triton.cdiv(B-C,128)](D,E,_local__panels_mass,G,A,B-C,D.shape[1],E.shape[1],C,_local__panels_lam,_local__panels_c,128,128,32,num_warps=8,enable_fp_fusion=False);F.append(G)
	return F,A
@triton.jit
def diagonal(b,k,i,d,f,c,h:A,j:A,g:A,e:A,a:A):t=X(0);m=I(0,a);l=H(b+t*j+m,m<j,0);q=H(k+t);s=A8(1.-tl.max(l*l,0)/(1.+e),.0);n=g+q*s;r=(D(q)*l/n).to(E);u=r.to(K)*n;o=u.to(E);p=(u-o.to(K)).to(E);Y(i+t*j+m,r,m<j);Y(d+t*j+m,o,m<j);Y(f+t*j+m,p,m<j);Y(c+t,1./n)
def Aa(_local_AC_z,_local_AC_labels,_local_AC_mass,_local_AC_lam,_local_AC_jit,_local_AC_c,_local_AC_cc):
	I=_local_AC_jit;H=_local_AC_c;G=_local_AC_cc;D=_local_AC_z;B,P=D.shape;C=len(G);J=U((B,C),device=D.device);radial[triton.cdiv(B,32),triton.cdiv(C,64)](D,G,J,B,C,P,32,64,num_warps=4,enable_fp_fusion=False);K=AM(G,G).double();K.diagonal().add_(I/H);E=U((B,C),device=D.device,dtype=d);L=c(E);M=c(E);N=U(B,device=D.device);diagonal[B,](J,_local_AC_mass,E,L,M,N,B,C,_local_AC_lam/H,I/H,triton.next_power_of_2(C),num_warps=4,enable_fp_fusion=False);F=(s(E.T,L,out_dtype=k)+s(E.T,M,out_dtype=k)).double()+K;F=(F+F.T)*.5;Q,R=f.cholesky_ex(F.float());O=f.solve_triangular(Q,o(C,device=D.device),upper=False);A=(O.T@O).double()
	for S in range(2):A=2*A-A@(F@A);A=(A+A.T)*.5
	return E,(A,N)
def Ab(_local__coreset_z,_local__coreset_labels):C=_local__coreset_labels;A=_local__coreset_z;B=AT(A,C,32);D=torch.arange(10,device=A.device).repeat_interleave(32);H=A@B.T-.5*B.square().sum(1)[None];M=H.masked_fill(C[:,None]!=D[None],float('-inf'));N,O=M.max(1);P=H.masked_fill(C[:,None]==D[None],float('-inf')).max(1).values;Q=N-P;I=Q.argsort(stable=True);J,K=I[:5120],I[5120:];L=F.one_hot(O[K],len(B)).float();E=L.sum(0);G=L.T@A[K]/E.clamp_min(1)[:,None];G=torch.where((E>0)[:,None],G,B);R=i([A[J],G]);S=i([C[J],D]);T=i([p(5120,device=A.device),E]);return R,S,T,B
def Ac(_local__fit_za,_local__fit_labels,_local__fit_y,_local__fit_c,_local__fit_lam,_local__fit_jit):
	J=_local__fit_lam;I=_local__fit_c;H=_local__fit_labels;G=_local__fit_za;K=len(H);B,L,A,Q=Ab(G[:K],H);C=len(B);M=F.one_hot(L.long(),10).float()*A.sqrt()[:,None];R=i([B,G[K:]]);N=j.current_stream();D=j.Stream();D.wait_stream(N)
	with j.stream(D):S,T=Aa(B,L,A,J,_local__fit_jit,I,Q)
	O,P=AN(R);U,V=AZ(O,P,C,A,J,I);N.wait_stream(D);E=AR(AV(U,V,C),i([M,M[:,:6]],1),5,S,T);E=(E*A.sqrt()[:,None]).contiguous();return AQ(O,P,C,E)[:,:10]

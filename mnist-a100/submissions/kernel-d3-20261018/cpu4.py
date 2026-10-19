
import triton,triton.language as tl
from triton.language import constexpr as G,float16 as A4,float32 as A2,float64 as Af,int32 as C,int64 as Ag,program_id as X,arange as I,load as H,sum as A0,full as A7,store as Y,sqrt as K,dot as A1,cdiv as N,where as A3,exp as L,constexpr as A,maximum as A5,float16 as D,max as O,float32 as E,trans as A6
from torch import empty as U,cat as i,eye as o,mm as s,empty_like as c,int32 as S,ones as p,float16 as f,float32 as n,zeros_like as q,linalg as k,cuda as j,bool as r
@triton.jit
def R(q,r,U,e,t:G,w:G,u:G,s:G,v:G):
	C=X(0);R=I(0,128);O=I(0,s);P=H(U+C*u+R,R<u,0);g=H(r+P[:,None]*t+O[None,:],(R[:,None]<u)&(O[None,:]<t),0);T=A0(g,0)/u;m=A3(R[:,None]<u,g-T[None,:],.0).to(A4);x=A1(A6(m),m)+A3(O[:,None]==O[None,:],v,.0);B=H(q+C//w*t+O,O<t,0)-T;o=m.to(A2);k=A0(A0(o*o,0),0);L=v-2*(7.62939453125e-06/.9999923706054688)*(k+v);p=A7((s,),.0,A2);h,f=B,B;a=A0(h*h,0);Q=a;j=0;c=1.;d=.0;J=1.;E=L;l=.0;S=.0;N=float('inf');M=L>0
	while(j<32)&(a>Q*1e-12)&((j<4)|(N-S>.01*A5(S,1e-30))|(N<S)|(N!=N)|(S!=S)|~M):
		A=A0(x*f[None,:],1);F=A0(f*A,0);z=a/A5(F,1e-30)
		if j>0:K=1.-c*E;M&=K>0;E=L+d*E/A5(K,1e-30);J=1.+d*J
		N=S+a/E*(2*l+J/E);S+=a*z*(2*l+z*J);l+=z*J;p+=z*f;h-=z*A;b=A0(h*h,0);D=b/A5(a,1e-30);f=h+f*D;a=b;c=z;d=D;j+=1
	n=v*v*A0(p*p,0);i=v*v*(S+N)*.5;y=M&(N>=S)&(N-S<=.01*A5(S,1e-30));Y(e+C,A3(y,i,n))
@triton.jit
def B(d,h,b:G,a:G,c:G):
	f=I(0,a);g=(f[:,None]<b)&(f[None,:]<b);o=H(d+f[:,None]*b+f[None,:],g,.0);e=((f[:,None]==f[None,:])&g).to(A2);j=K(A0(A0(o*o,0),0));p,q=o/j,e
	for l in range(c):m=1.5*e-.5*A1(q,p,input_precision='tf32x3');p,q=A1(p,m,input_precision='tf32x3'),A1(q,m,input_precision='tf32x3')
	i=p*K(j);k=K(A0(A0(i*i,0),0));p,q=i/k,e
	for l in range(c):m=1.5*e-.5*A1(q,p,input_precision='tf32x3');p,q=A1(p,m,input_precision='tf32x3'),A1(q,m,input_precision='tf32x3')
	o=A1(i,p*K(k),input_precision='tf32x3');o=(o+A6(o))*.5;n=A0(A0(o*e,0),0);o=.98*o*(b/A5(n,1e-12))+.02*e;Y(h+f[:,None]*b+f[None,:],o,g)
M=4096
@triton.jit
def Z(f,l,i,d,b:G,a:G):j=X(0)*a+I(0,a);c=I(0,16);m=H(f+j,j<b,0);g=H(l+j,j<b,0)&(j<b);e=(c[:,None]==m[None,:])&g[None,:];k=tl.cumsum(e.to(C),1);h=A0(A3(e,k,0),0)-1;Y(i+j,h,j<b);Y(d+X(0)*16+c,A0(e.to(C),1))
@triton.jit
def a(k,s,q,m,o,j,c:G,b:G,a:G,d:G,e:G):
	g=X(0);h=I(0,e);i=I(0,16);n=H(m+h[:,None]*16+i[None,:],h[:,None]<d,0);f=A0(A3(h[:,None]<g,n,0),0);r=g*a+I(0,a);t=H(k+r,r<c,0).to(C);l=H(s+r,r<c,0)&(r<c);p=H(q+r,r<c,0)+tl.gather(f,t,0);Y(o+t*b+p,r,l&(p<b))
	if g==0:Y(j+i,A0(n,0),i<10)
def t(labels,valid,cap):E=valid;A=labels;B=len(A);D=512;C=triton.cdiv(B,D);F=U(B,device=A.device,dtype=S);G=U(C,16,device=A.device,dtype=S);H=torch.full((10,cap),B,device=A.device,dtype=torch.long);I=U(10,device=A.device,dtype=S);Z[C,](A,E,F,G,B,D,num_warps=4);a[C,](A,E,F,G,H,I,B,cap,D,C,triton.next_power_of_2(C),num_warps=4);return H,I
@triton.jit
def b(t,u,p,i,m,l,d:G,e:G,f:G,c:G,g:G,b:G,a:G):
	h=X(0)
	if(X(1)*b<H(m+h))&(X(2)*a<H(i+h)):n=X(1)*b+I(0,b);q=X(2)*a+I(0,a);j=I(0,64);o=H(p+h*g+n,n<g,f);r=H(t+o[:,None]//e*d+j[None,:],(o[:,None]<f)&(j[None,:]<d),.0);s=H(u+(h*c+q[None,:])*d+j[:,None],(q[None,:]<c)&(j[:,None]<d),.0);k=A5(A0(r*r,1)[:,None]+A0(s*s,0)[None,:]-2*A1(r,s,input_precision='tf32x3'),.0);k=A3(q[None,:]<H(i+h),k,float('inf'));Y(l+o[:,None]*c+q[None,:],k,(o[:,None]<f)&(q[None,:]<c))
def AC(zq,zv,yv,valid,cls):B=cls;A,C=B.shape;D,F=t(yv,valid,M);G,H=t(B.flatten(),p(B.numel(),device=B.device,dtype=r),A);I=i([zv,q(zv[:1])])[D].contiguous();E=torch.full((A*C,M),float('inf'),device=zq.device,dtype=f);d(b,(10,triton.cdiv(A,32),triton.cdiv(M,128)),4,zq,I,G,F.clamp_max(M),H,E,zq.shape[1],C,A*C,M,A,32,128);L,J=AU(E,Q);K=D[B.flatten()[:,None],J].reshape(A,C,Q);return K
def AD(zq,zv,yv,valid,cls):B,A=cls.shape;E=AC(zq,zv,yv,valid,cls);C=zq.shape[1];D=U(B*A,device=zq.device);d(R,(B*A,),4,zq,zv,E,D,C,A,Q,triton.next_power_of_2(C),AI);return D.reshape(B,A)
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
	g=e.to(A4);i=(e-g.to(A2)).to(A4);Y(j+k*(2*b)+f,g,k<c);Y(j+k*(2*b)+f+b,i,k<c)
import gc,math,torch,torch.nn.functional as F
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False
P,y,Ah,Ai,AF=.02,.1,256,.02,.6
Aj=7
AG,AH=.2,512
Q,AI,AJ,AK,AL,Ak=70,8.,2.,.15,5,2048
Al,Am,An,u=5000,1000,7,3.
v={}
def z(a):return k.cholesky_ex(a)[0]
def A8(x):return x*(x.shape[1]**.5/x.norm(dim=1,keepdim=True).clamp_min(1e-06))**AF
def A9(a,b,shift=.0):A=a.square().sum(1,keepdim=True);B=b.square().sum(1,keepdim=True);C=i([a,A,torch.ones_like(A)],1);D=i([2*P*b,torch.full_like(B,-P),shift-P*B],1).T.contiguous();return C,D
def AM(a,b):A,B=A9(a,b);return(A@B).exp_()
def AN(z):C,D=A9(z,z,u);A,B=C.half(),D.half();H,I=(C-A.float()).half(),(D-B.float()).half();E,J=i([A,A,H],1),i([B,I,B],0);G=-E.shape[1]%16;return F.pad(E,(0,G)),F.pad(J,(0,0,0,G))
def AO(d,t):A=d.min(1,keepdim=True).values;return torch.softmax(-(d-A)/(t*A).clamp_min(1e-12),1)
def AP(tx,y):D,E=tx.shape;G=y.sum(0);A=(y.T@tx*(G*D).rsqrt()[:,None]).T.double();C=1.-1./D+.001;F=o(E,device=tx.device,dtype=torch.float64);H=F[:10,:10]-A.T@A/C;I=z(H);J=torch.cholesky_solve(A.T,I);B=F/C+3./C**2*(A@J);B=(B*(E/B.trace())).float();K=z(B);return K
def AA(train_x,train_y,test_x):
	N=train_y;O=train_x;P=O;H=N;B=math.exp(u);X,Y,d=y*B,.0001*B,B*(1.+y)
	with torch.no_grad():E,Z=A8(P.float()),A8(test_x.float());G=len(E);I=F.one_hot(H.long(),10).float();J=AP(P,I);e=j.current_stream();f=j.Stream();J=AY(O,E,I,N,J);A=i([E,Z])@J;Q=A[:G].square().sum(1).mean();R=A.norm(dim=1,keepdim=True).clamp_min(1e-06);K=Q.sqrt();A=i([A*(K/R),.5*K*(R/K).log()],1);A=A*(Q/A[:G].square().sum(1).mean()).sqrt();L=Ac(A,H,I,B,X,Y);S=L.topk(2,1).values;T=S[:,0]-S[:,1];U=T>=AG;M=L.argmax(1);C=T.topk(AH,largest=False).indices;D=L[C]/B;V=D.topk(AL,1).indices;a=i([H.long(),M]);b=i([p(G,dtype=r,device=E.device),U]);c=AO(AD(A[G:][C],A,a,b,V),AJ);D=D+AK*q(D).scatter_(1,V,c);W=M.clone();W[C]=torch.where(U[C],M[C],D.argmax(1));return W
def rfm_d3(train_x,train_y,test_x):
	D=test_x;E=train_y;B=train_x;F=tuple(B.shape),tuple(D.shape),B.dtype,E.dtype,str(B.device)
	if F not in v:
		A=[B.clone(),E.clone(),D.clone()];G=j.Stream();G.wait_stream(j.current_stream())
		with j.stream(G):
			for I in range(2):AA(*A)
		j.current_stream().wait_stream(G);C=j.CUDAGraph()
		with j.graph(C):H=AA(*A)
		v[F]=C,A,H;gc.collect();gc.freeze()
		for J in range(64):C.replay()
		j.synchronize()
	C,A,H=v[F];A[0].copy_(B);A[1].copy_(E);A[2].copy_(D);C.replay();return H.clone()
AB,x=128,4
@triton.jit
def g(D,q,j,A:G,z:G,y:G,v:G,u:G,t:G,s:G,C:G,x:G,w:G,B:G):
	l=X(0)*t+I(0,t);g=I(0,64);f=I(0,s);c=I(0,16);h:G=2*(v+2);F=H(D+(z+l[:,None])*y+g[None],(l[:,None]<A)&(g[None]<v),0);n=H(D+(z+l)*y+v,l<A,0).to(A2)+H(D+(z+l)*y+h+v,l<A,0).to(A2);E=A7((t,16),0,A2);b=N(z,s*C)
	for o in range(b):e=(X(1)*b+o)*s+f;a=H(D+e[None]*y+g[:,None],(g[:,None]<v)&(e[None]<z),0);d=A1(F,a);m=H(D+e*y+v,e<z,0).to(A2)+H(D+e*y+h+v,e<z,0).to(A2);i=(B+2*w*d-w*(n[:,None]+m[None])).to(A4).to(A2);r=L(i).to(A4);p=H(q+e[:,None]*u+c[None],(e[:,None]<z)&(c[None]<u),0);k=A1(r,p);E=tl.inline_asm_elementwise('add.rn.f32 $0, $1, $2;',constraints='=f,f,f',args=[E,k],dtype=A2,is_pure=True,pack=1)
	Y(j+(X(1)*A+l[:,None])*u+c[None],E,(l[:,None]<A)&(c[None]<u))
def AQ(a,b,n,w):B=w.abs().amax(0,keepdim=True).clamp_min(1e-30);D=(w/B).half();A=len(a)-n;C=U((x,A,w.shape[1]),device=w.device);g[triton.cdiv(A,AB),x](a,D,C,A,n,a.shape[1],AE,w.shape[1],AB,64,x,False,P,u,num_warps=4);E=C.sum(0)*B;return E
def w(phi,inv,r,rh=None):
	C=inv[0];B,A=r.shape
	if rh is None:rh=U((B,2*A),device=r.device,dtype=f);J[triton.cdiv(B*A,1024),](r,rh,B,A,False,1024,num_warps=4)
	E=s(phi.T,rh,out_dtype=n);H=C@(E[:,:A].double()+E[:,A:].double());F=U((len(C),2*A),device=r.device,dtype=f);J[triton.cdiv(len(C)*A,1024),](H,F,len(C),A,False,1024,num_warps=4);I=s(phi,F,out_dtype=n);G=c(r);D=U((triton.cdiv(B,256),16),device=r.device);d(m,(len(D),),4,I,r,G,D,inv[1],B,A,256);return G,D
def AR(op,t,steps,phi,inv):
	Q=steps;K=inv;L=phi;E,F=t.shape;B=triton.cdiv(E,256);M=U((B,16),device=t.device);N,I=q(t),t.clone();A,T=w(L,K,I);C,G=T.sum(0),A.half();O=U((E,2*F),device=t.device,dtype=f)
	for R in range(Q):
		S=op.mv(G,R%2==1);e[B,](A,S,M,E,F,256,num_warps=4);d(h,(B,),4,N,I,A,S,M,C,O,E,F,256,B,triton.next_power_of_2(B))
		if R+1==Q:break
		P,D=w(L,K,I,O);H,G=c(A),c(A,dtype=f);J=c(C);d(V,(B,),4,P,A,D,C,J,H,G,E,F,256,len(D),triton.next_power_of_2(len(D)));A,C=H,J
	P,D=w(L,K,I,O);H,G=c(A),c(A,dtype=f);J=c(C);d(V,(B,),4,P,A,D,C,J,H,G,E,F,256,len(D),triton.next_power_of_2(len(D)));return N+H*(C/M.sum(0));return N
@triton.jit
def h(x,t,j,h,r,p,u,c:G,b:G,a:G,d:G,e:G):s,i=I(0,e),I(0,16);l=A0(H(r+s[:,None]*16+i[None],s[:,None]<d,0),0);g=H(p+i,i<b,0)/l;v=X(0)*a+I(0,a);q=v[:,None]*b+i[None];o=(v[:,None]<c)&(i[None]<b);y,w=H(x+q,o,0),H(t+q,o,0);k,f=H(j+q,o,0),H(h+q,o,0);w=w-f*g[None];Y(x+q,y+k*g[None],o);Y(t+q,w,o);m=w.to(A4);n=(w-m.to(A2)).to(A4);Y(u+v[:,None]*(2*b)+i[None],m,o);Y(u+v[:,None]*(2*b)+i[None]+b,n,o)
def AS(a):A=c(a);d(B,(1,),4,a,A,len(a),triton.next_power_of_2(len(a)),7);return A
@triton.jit
def T(f,a,e,b:G,c:G,d:G,g:G):i=X(0);j,m=i//c,i%c;h=I(0,g);l=I(0,64);q=H(f+((j*d+h[:,None])*c+m)*64+l[None,:],h[:,None]<d,0);p=A0(q,0);k=A0(A3(l==b,p,0),0);o=H(a+i*b+l,l<b,0);n=A3(k>0,p/A5(k,1),o);Y(e+i*b+l,n,l<b)
def AT(z,labels,mc=32):
	A=mc;D=1400;F,J=t(labels,p(len(z),device=z.device,dtype=r),D);G=i([z,q(z[:1])])[F].contiguous();I=F<len(z);B=G[:,:A].reshape(10*A,-1).contiguous();C=triton.cdiv(D,128);E=U((10,C,A,64),device=z.device,dtype=n)
	for K in range(2):H=c(B);d(W,(10,C),4,G,I,B,E,E,D,z.shape[1],A,triton.next_power_of_2(A),C,128,False,True);d(T,(10*A,),4,E,B,H,z.shape[1],A,C,triton.next_power_of_2(C));B=H
	return B
@triton.jit
def l(i,s,k,c:G,b:G,a:G):
	p=X(0);g=I(0,a);r=H(i+p*c+g,g<c,0);f=r.to(tl.uint16,bitcast=True).to(C)&32767;m=A7((),0,C)
	for q in range(15):e=m|1<<14-q;h=A0(((f<e)&(g<c)).to(C),0);m=A3(h<b,e,m)
	d=(f<m)&(g<c);j=(f==m)&(g<c);n=b-A0(d.to(C),0);l=d|j&(tl.cumsum(j.to(C),0)<=n);o=tl.cumsum(l.to(C),0)-1;Y(s+p*b+o,r,l);Y(k+p*b+o,g,l)
def AU(distance,k=70):A=distance;B,C=A.shape;D=U((B,k),device=A.device,dtype=A.dtype);E=U((B,k),device=A.device,dtype=torch.int64);l[B,](A,D,E,C,k,triton.next_power_of_2(C),num_warps=8);return D,E
@triton.jit
def m(l,m,k,h,g,c:G,b:G,a:G):n=X(0)*a+I(0,a);f=I(0,16);i=(n[:,None]<c)&(f[None,:]<b);j=H(m+n[:,None]*b+f[None,:],i,0);d=H(l+n[:,None]*(2*b)+f[None,:],i,0);e=H(l+n[:,None]*(2*b)+b+f[None,:],i,0);o=H(g+n,n<c,0);p=j*o[:,None]-(d+e);Y(k+n[:,None]*b+f[None,:],p,i);Y(h+X(0)*16+f,A0(j*p,0))
class AV:
	def __init__(A,pans,pp,n):A.pans,A.pp,A.n=pans,pp,n
	def mv(A,rhs,rev):B=rhs;C=A.n//A.pp;D=2 if C==4 else 1;F=U((C*C*D,A.pp,B.shape[1]),device=B.device);E=U((A.n,B.shape[1]),device=B.device);d(_packed_mv,(triton.cdiv(A.pp,128),D,C*C),4,A.pans,B,F,A.n,A.pp,B.shape[1],C,128,128,D,rev);d(_packed_reduce,(triton.cdiv(E.numel(),512),),4,F,E,A.n,A.pp,B.shape[1],C,D,512);return E
def AW(a,b):return torch.bmm(a.half(),b.half(),out_dtype=n)
def AX(m,rawx):A=m.double();E=rawx[:48].T.double();C=A@E;D=A@C;F=A@D;G=1e-07*A.square().sum()/48;H=E.T@C;I=C.T@D;J=D.T@F;B=J+G*I+G.square()*(H+o(48,device=m.device,dtype=torch.float64)*(1e-07*H.trace()/48));B=(B+B.T)*.5;K,M=k.cholesky_ex(B);L=k.solve_triangular(K,F.T,upper=False).T;return L.float()
def AY(raw,x,y,labels,initial):B=initial;J=k.cholesky_ex(.5*o(B.shape[0],device=x.device)+.5*(B@B.T))[0];D=B;R,K=x.shape;E=2;L,A,F,M=AZ(x,labels);N=o(K,device=x.device);A=.6*A+.4*torch.cholesky_inverse(J);G,S=k.cholesky_ex(A);O=torch.cholesky_inverse(G);H=(x[:2048][None]-L[:,None]).contiguous();I=-AW(H,O);P=torch.where(F>0,(F/M.float().repeat_interleave(E)).log(),float('-inf'));C=.5*(H*I).sum(2)-G.diagonal(dim1=1,dim2=2).log().sum(1)[:,None]+P[:,None];C=C.reshape(10,E,2048).permute(2,0,1);A=Ae(I,C);Q=.8*AS(A)+.196*(D@D.T)+.004*N;return AX(Q,x)
@triton.jit
def W(A,z,q,w,o,r:G,s:G,t:G,v:G,u:G,p:G,x:G,y:G):
	a,D=X(0),X(1);h=D*p+I(0,p);c=I(0,64);b=I(0,v);l=H(A+(a*r+h[:,None])*s+c[None,:],(h[:,None]<r)&(c[None,:]<s),0);E=H(q+(a*t+b[:,None])*s+c[None,:],(b[:,None]<t)&(c[None,:]<s),0)
	if x:m,F=l.to(A4),E.to(A4);n,G=(l-m.to(A2)).to(A4),(E-F.to(A2)).to(A4);i=A1(m,A6(F))+A1(n,A6(F))+A1(m,A6(G));i=i-.5*A0(E*E,1)[None,:]
	else:g=(-.5*A0(E*E,1)).to(A4).to(A2);i=(A1(l.to(A4),A6(E.to(A4)))+g[None,:]).to(A4).to(A2)
	i=A3(b[None,:]<t,i,float('-inf'));B=tl.argmax(i,1,tie_break_left=True)
	if x:Y(o+a*r+h,B,h<r)
	if y:k=H(z+a*r+h,h<r,0);f=((B[None,:]==b[:,None])&k[None,:]).to(A4);C=A3(c[None,:]==s,k[:,None].to(A2),l);d=C.to(A4);e=(C-d.to(A2)).to(A4);j=A1(f,d)+A1(f,e);Y(w+((a*u+D)*t+b[:,None])*64+c[None,:],j,b[:,None]<t)
@triton.jit
def mom2(j,i,c,g,e:G,f:G,d:G,h:G):
	s=X(0);p,q=s//2,s%2;r=I(0,64);u=I(0,d);k=A7((64,64),.0,A2);l=A7((64,64),.0,A2);n=A7((64,64),.0,A2);o=N(e,d*h)
	for A in range(o):z=(X(1)*o+A)*d+u;a=H(i+p*e+z,z<e,0);y=H(c+p*e+z,z<e,-1)==q;b=H(j+(p*e+z[:,None])*f+r[None,:],(z[:,None]<e)&(r[None,:]<f),0);m=A3(r[None,:]==f,1.,b);t=m.to(A4);v=(m-t.to(A2)).to(A4);w=A3((a&y)[:,None],t,.0);x=A3((a&y)[:,None],v,.0);k=A1(A6(w),t,k);l=A1(A6(w),v,l);n=A1(A6(x),t,n)
	Y(g+((s*h+X(1))*64+r[:,None])*64+r[None,:],k+n+l)
def AZ(x,labels):
	I,A=x.shape;C=1400;J=64;K,Q=t(labels,p(I,device=x.device,dtype=r),C);F=i([x,q(x[:1])])[K].contiguous();L=K<I;E=F[:,:2].reshape(20,A).contiguous();D=triton.cdiv(C,J);M=U((10,D,2,64),device=x.device);N=U((10,C),device=x.device,dtype=S)
	for O in range(3):
		d(W,(10,D),4,F,L,E,M,N,C,A,2,16,D,J,True,O<2)
		if O<2:P=c(E);d(T,(20,),4,M,E,P,A,2,D,triton.next_power_of_2(D));E=P
	B=U((20,8,64,64),device=x.device);d(mom2,(20,8),4,F,L,N,B,C,A,128,8);B=B.sum(1);G=B[:,A,A];H=B[:,:A,A]/G.clamp_min(1)[:,None];R=B[:,:A,:A]/G.clamp_min(1)[:,None,None]-H[:,:,None]*H[:,None,:];return H,R,G,Q
@triton.jit
def radial(h,c,f,e:G,g:G,d:G,a:G,b:G):o=X(0)*a+I(0,a);k=X(1)*b+I(0,b);n=I(0,64);r=H(h+o[:,None]*d+n[None,:],(o[:,None]<e)&(n[None,:]<d),0);i=H(c+k[:,None]*d+n[None,:],(k[:,None]<g)&(n[None,:]<d),0);s=A0(r*r,1);j=A0(i*i,1);p=i*.04;m=A1(r,A6(p),input_precision='tf32x3');l=m-s[:,None]*.02-j[None,:]*.02;q=L(l);Y(f+o[:,None]*g+k[None,:],q,(o[:,None]<e)&(k[None,:]<g))
@triton.jit
def diagonal(b,k,i,d,f,c,h:A,j:A,g:A,e:A,a:A):t=X(0);m=I(0,a);l=H(b+t*j+m,m<j,0);q=H(k+t);s=A5(1.-O(l*l,0)/(1.+e),.0);n=g+q*s;r=(K(q)*l/n).to(D);u=r.to(E)*n;o=u.to(D);p=(u-o.to(E)).to(D);Y(i+t*j+m,r,m<j);Y(d+t*j+m,o,m<j);Y(f+t*j+m,p,m<j);Y(c+t,1./n)
def Aa(_local_AC_z,_local_AC_labels,_local_AC_mass,_local_AC_lam,_local_AC_jit,_local_AC_c,_local_AC_cc):
	I=_local_AC_jit;H=_local_AC_c;G=_local_AC_cc;D=_local_AC_z;B,P=D.shape;C=len(G);J=U((B,C),device=D.device);d(radial,(triton.cdiv(B,32),triton.cdiv(C,64)),4,D,G,J,B,C,P,32,64);K=AM(G,G).double();K.diagonal().add_(I/H);E=U((B,C),device=D.device,dtype=f);L=c(E);M=c(E);N=U(B,device=D.device);d(diagonal,(B,),4,J,_local_AC_mass,E,L,M,N,B,C,_local_AC_lam/H,I/H,triton.next_power_of_2(C));F=(s(E.T,L,out_dtype=n)+s(E.T,M,out_dtype=n)).double()+K;F=(F+F.T)*.5;Q,R=k.cholesky_ex(F.float());O=k.solve_triangular(Q,o(C,device=D.device),upper=False);A=(O.T@O).double()
	for S in range(2):A=2*A-A@(F@A);A=(A+A.T)*.5
	return E,(A,N)
def Ab(_rank__coreset_z,_rank__coreset_labels):E=_rank__coreset_labels;A=_rank__coreset_z;C,F=A.shape;D=AT(A,E,32);H=U(C,device=A.device);I=U(C,device=A.device,dtype=S);d(_priorities,(triton.cdiv(C,32),),8,A,D,E,H,I,C,F);J=H.argsort(stable=True);K,G=J[:5120],J[5120:];B=triton.cdiv(len(G),128);L=U((10,B,32,64),device=A.device);d(_moments,(10,B),4,A,G,I,L,len(G),F,B);M=c(D);N=U(320,device=A.device);d(_finish,(320,),4,L,D,M,N,F,B,triton.next_power_of_2(B));O=torch.arange(10,device=A.device).repeat_interleave(32);P=i([A[K],M]);Q=i([E[K],O]);R=i([p(5120,device=A.device),N]);return P,Q,R,D
def Ac(_local__fit_za,_local__fit_labels,_local__fit_y,_local__fit_c,_local__fit_lam,_local__fit_jit):
	J=_local__fit_lam;I=_local__fit_c;H=_local__fit_labels;G=_local__fit_za;K=len(H);B,L,A,Q=Ab(G[:K],H);C=len(B);M=F.one_hot(L.long(),10).float()*A.sqrt()[:,None];R=i([B,G[K:]]);N=j.current_stream();D=j.Stream();D.wait_stream(N)
	with j.stream(D):S,T=Aa(B,L,A,J,_local__fit_jit,I,Q)
	O,P=AN(R);U,V=Ad(O,P,C,A,J,I);N.wait_stream(D);E=AR(AV(U,V,C),i([M,M[:,:6]],1),5,S,T);E=(E*A.sqrt()[:,None]).contiguous();return AQ(O,P,C,E)[:,:10]
@triton.jit
def _priorities(g,b,d,f,a,e:A,c:A):q=X(0)*32+I(0,32);j=I(0,512);k=I(0,64);s=H(g+q[:,None]*c+k[None,:],(q[:,None]<e)&(k[None,:]<c),0);i=H(b+j[:,None]*c+k[None,:],(j[:,None]<320)&(k[None,:]<c),0);r=A1(s,A6(i),input_precision='tf32x3')-.5*A0(i*i,1)[None,:];l=H(d+q,q<e,-1);n=(j[None,:]//32==l[:,None])&(j[None,:]<320);o=A3(n,r,float('-inf'));h=tl.argmax(o,1,tie_break_left=True);p=O(A3(~n&(j[None,:]<320),r,float('-inf')),1);m=O(o,1)-p;Y(f+q,m,q<e);Y(a+q,h,q<e)
@triton.jit
def _moments(g,c,a,f,d:A,b:A,e:A):l=X(0);i=X(1);q=i*128+I(0,128);k=I(0,64);j=I(0,32);n=H(c+q,q<d,0);h=H(a+n,q<d,-1);s=H(g+n[:,None]*b+k[None,:],(q[:,None]<d)&(k[None,:]<b),0);s=A3(k[None,:]==b,1.,s);p=((h[None,:]==l*32+j[:,None])&(q[None,:]<d)).to(D);m=s.to(D);o=(s-m.to(E)).to(D);r=A1(p,m)+A1(p,o);Y(f+((l*e+i)*32+j[:,None])*64+k[None,:],r)
@triton.jit
def _finish(f,b,e,g,c:A,d:A,a:A):i=X(0);l=i//32;m=i%32;h=I(0,a);k=I(0,64);q=H(f+((l*d+h[:,None])*32+m)*64+k[None,:],h[:,None]<d,0);p=A0(q,0);j=A0(A3(k==c,p,.0),0);o=H(b+i*c+k,k<c,0);n=A3(j>0,p/A5(j,1.),o);Y(e+i*c+k,n,k<c);Y(g+i,j)
@triton.jit
def _construct(a,b,l,i,h:A,j:A,f:A,k:A,g:A,e:A,c:A,d:A):
	t=X(2);q=((K((8*t+1).to(E))-1)*.5).to(tl.int32);w=t-q*(q+1)//2;x=X(0)*c+I(0,c);r=X(1)*d+I(0,d);v=w*j+x;p=q*j+r;u=I(0,32);n=A7((c,d),.0,E)
	for y in range(N(f,32)):s=y*32+u;m=H(a+v[:,None]*f+s[None,:],(x[:,None]<j)&(s[None,:]<f),0);o=H(b+s[:,None]*k+p[None,:],(s[:,None]<f)&(r[None,:]<j),0);n=A1(m,o,n)
	z=L(n.to(D).to(E)).to(D).to(E);B=H(l+v,x<j,0);A=H(l+p,r<j,0);z=z*K(B[:,None]*A[None,:]);z=A3(v[:,None]==p[None,:],e*B[:,None]+g,z).to(D);Y(i+(t*j+x[:,None])*j+r[None,:],z,(x[:,None]<j)&(r[None,:]<j))
@triton.jit
def _packed_mv(g,i,f,d:A,h:A,c:A,e:A,b:A,a:A,k:A,j:A):
	t=X(2);y=t//e;p=t%e;u=A5(y,p);w=tl.minimum(y,p);x=u*(u+1)//2+w;z=X(0)*b+I(0,b);v=I(0,a);q=I(0,16);s=N(h,a*k);m=A7((b,16),.0,E)
	for A in range(s):
		o=s-1-A if j else A;r=(X(1)*s+o)*a+v
		if y>p:l=H(g+(x*h+r[None,:])*h+z[:,None],(z[:,None]<h)&(r[None,:]<h),0)
		else:l=H(g+(x*h+z[:,None])*h+r[None,:],(z[:,None]<h)&(r[None,:]<h),0)
		n=H(i+(p*h+r[:,None])*c+q[None,:],(r[:,None]<h)&(q[None,:]<c),0);m=A1(l,n,m)
	Y(f+((t*k+X(1))*h+z[:,None])*c+q[None,:],m,(z[:,None]<h)&(q[None,:]<c))
@triton.jit
def _packed_reduce(f,e,c:A,g:A,b:A,d:A,h:A,a:A):
	k=X(0)*a+I(0,a);n=k//b;j=k%b;m=n//g;l=n%g;p=A7((a,),.0,E)
	for i in range(d):
		for o in range(h):p+=H(f+(((m*d+i)*h+o)*g+l)*b+j,n<c,0)
	Y(e+k,p,n<c)
def Ad(a,b,n,mass,lam,c):A=n//4;assert A*4==n;B=U((10,A,A),device=a.device,dtype=f);d(_construct,(triton.cdiv(A,128),triton.cdiv(A,128),len(B)),8,a,b,mass,B,n,A,a.shape[1],b.shape[1],lam,c,128,128);return B,A
@triton.jit
def _information_kernel(c,d,f,e:A,b:A,g:A,h:A,i:A,a:A):x=X(0)*a+I(0,a);m=I(0,16);n=I(0,64);p=H(d+x[:,None]*g+m[None,:]*h,(x[:,None]<e)&(m[None,:]<10),0);A=H(d+x[:,None]*g+m[None,:]*h+i,(x[:,None]<e)&(m[None,:]<10),0);t=A5(p,A);u=L(p-t);v=L(A-t);B=u+v;j=u/B;k=v/B;y=A3(m[None,:]<10,(t+tl.log(B))*.25,float('-inf'));w=L(y-O(y,1)[:,None]);w=w/A0(w,1)[:,None];s=(x[:,None,None]<e)&(m[None,:,None]<10)&(n[None,None,:]<b);q=H(c+(2*m[None,:,None]*e+x[:,None,None])*b+n[None,None,:],s,0);r=H(c+((2*m[None,:,None]+1)*e+x[:,None,None])*b+n[None,None,:],s,0);z=q*j[:,:,None]+r*k[:,:,None];l=A0(z*w[:,:,None],1);o=(z-l[:,None,:])*K(w)[:,:,None];o=tl.reshape(o,(a*16,64)).to(D);C=A1(A6(o),o);Y(f+(X(0)*b+n[:,None])*b+n[None,:],C,(n[:,None]<b)&(n[None,:]<b))
def Ae(gradient,logits):B=logits;A=gradient;E=B.shape[0];C=A.shape[-1];D=U((triton.cdiv(E,8),C,C),device=A.device);d(_information_kernel,(len(D),),4,A,B,D,E,C,*B.stride(),8);F=D.sum(0);return F
def d(kernel,grid,warps,*A):return kernel[grid](*A,num_warps=warps,enable_fp_fusion=False)

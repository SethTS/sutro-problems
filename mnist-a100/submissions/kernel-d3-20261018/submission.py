from torch import float64 as u,empty as G,cat as O,eye as A0,mm as A5,empty_like as Y,int32 as j,ones as p,float16 as f,float32 as t,zeros_like as q,linalg as c,cuda as d,bool as A1
from triton import cdiv as L,next_power_of_2 as X
from triton.language import float64 as A6,int32 as n,program_id as I,arange as E,load as D,sum as H,full as o,store as M,sqrt as k,dot as Q,cdiv as w,where as P,exp as x,constexpr as A,maximum as U,float16 as N,max as r,float32 as C,trans as S
import triton,triton.language as tl
@triton.jit
def R(f,g,O,T,i:A,l:A,j:A,h:A,k:A):
	Y=I(0);W=E(0,128);G=E(0,h);w=D(O+Y*j+W,W<j,0);m=D(g+w[:,None]*i+G[None,:],(W[:,None]<j)&(G[None,:]<i),0);n=H(m,0)/j;Z=P(W[:,None]<j,m-n[None,:],.0).to(N);x=Q(S(Z),Z)+P(G[:,None]==G[None,:],k,.0);p=D(f+Y//l*i+G,G<i,0)-n;q=Z.to(C);y=H(H(q*q,0),0);a=k-2*(7.62939453125e-06/.9999923706054688)*(y+k);b=o((h,),.0,C);J,L=p,p;F=H(J*J,0);z=F;X=0;r=1.;c=.0;R=1.;V=a;d=.0;A=.0;B=float('inf');e=a>0
	while(X<32)&(F>z*1e-12)&((X<4)|(B-A>.01*U(A,1e-30))|(B<A)|(B!=B)|(A!=A)|~e):
		s=H(x*L[None,:],1);A0=H(L*s,0);K=F/U(A0,1e-30)
		if X>0:t=1.-r*V;e&=t>0;V=a+c*V/U(t,1e-30);R=1.+c*R
		B=A+F/V*(2*d+R/V);A+=F*K*(2*d+K*R);d+=K*R;b+=K*L;J-=K*s;u=H(J*J,0);v=u/U(F,1e-30);L=J+L*v;F=u;r=K;c=v;X+=1
	A1=k*k*H(b*b,0);A2=k*k*(A+B)*.5;A3=e&(B>=A)&(B-A<=.01*U(A,1e-30));M(T+Y,P(A3,A2,A1))
@triton.jit
def B(d,h,b:A,a:A,c:A):
	F=E(0,a);L=(F[:,None]<b)&(F[None,:]<b);A=D(d+F[:,None]*b+F[None,:],L,.0);I=((F[:,None]==F[None,:])&L).to(C);N=k(H(H(A*A,0),0));B,G=A/N,I
	for P in range(c):J=1.5*I-.5*_n(G,B);B,G=_n(B,J),_n(G,J)
	K=B*k(N);O=k(H(H(K*K,0),0));B,G=K/O,I
	for P in range(c):J=1.5*I-.5*_n(G,B);B,G=_n(B,J),_n(G,J)
	A=_n(K,B*k(O));A=(A+S(A))*.5;Q=H(H(A*I,0),0);A=.98*A*(b/U(Q,1e-12))+.02*I;M(h+F[:,None]*b+F[None,:],A,L)
i=4096
@triton.jit
def Z(f,l,i,d,b:A,a:A):A=I(0)*a+E(0,a);C=E(0,16);F=D(f+A,A<b,0);G=D(l+A,A<b,0)&(A<b);B=(C[:,None]==F[None,:])&G[None,:];J=tl.cumsum(B.to(n),1);K=H(P(B,J,0),0)-1;M(i+A,K,A<b);M(d+I(0)*16+C,H(B.to(n),1))
@triton.jit
def a(k,s,q,m,o,j,c:A,b:A,a:A,d:A,e:A):
	B=I(0);C=E(0,e);F=E(0,16);G=D(m+C[:,None]*16+F[None,:],C[:,None]<d,0);L=H(P(C[:,None]<B,G,0),0);A=B*a+E(0,a);J=D(k+A,A<c,0).to(n);N=D(s+A,A<c,0)&(A<c);K=D(q+A,A<c,0)+tl.gather(L,J,0);M(o+J*b+K,A,N&(K<b))
	if B==0:M(j+F,H(G,0),F<10)
def z(l,m,k):E=m;A=l;B=len(A);D=512;C=L(B,D);F=G(B,device=A.device,dtype=j);H=G(C,16,device=A.device,dtype=j);I=torch.full((10,k),B,device=A.device,dtype=torch.long);J=G(10,device=A.device,dtype=j);Z[C,](A,E,F,H,B,D,num_warps=4);a[C,](A,E,F,H,I,J,B,k,D,C,X(C),num_warps=4);return I,J
@triton.jit
def b(t,u,p,i,m,l,d:A,e:A,f:A,c:A,g:A,b:A,a:A):
	A=I(0)
	if(I(1)*b<D(m+A))&(I(2)*a<D(i+A)):L=I(1)*b+E(0,b);B=I(2)*a+E(0,a);C=E(0,64);F=D(p+A*g+L,L<g,f);G=D(t+F[:,None]//e*d+C[None,:],(F[:,None]<f)&(C[None,:]<d),.0);J=D(u+(A*c+B[None,:])*d+C[:,None],(B[None,:]<c)&(C[:,None]<d),.0);K=U(H(G*G,1)[:,None]+H(J*J,0)[None,:]-2*_n(G,J),.0);K=P(B[None,:]<D(i+A),K,float('inf'));M(l+F[:,None]*c+B[None,:],K,(F[:,None]<f)&(B[None,:]<c))
def AB(x,y,w,v,u):B=u;A,C=B.shape;D,F=z(w,v,i);G,H=z(B.flatten(),p(B.numel(),device=B.device,dtype=A1),A);I=O([y,q(y[:1])])[D].contiguous();E=torch.full((A*C,i),float('inf'),device=x.device,dtype=f);K(b,(10,L(A,32),L(i,128)),4,x,I,G,F.clamp_max(i),H,E,x.shape[1],C,A*C,i,A,32,128);N,J=AV(E,A2);M=D[B.flatten()[:,None],J].reshape(A,C,A2);return M
def AC(j,k,i,h,g):B,A=g.shape;E=AB(j,k,i,h,g);C=j.shape[1];D=G(B*A,device=j.device);K(R,(B*A,),4,j,k,E,D,C,A,A2,X(C),AI);return D.reshape(B,A)
AD=49
@triton.jit
def e(k,l,i,c:A,b:A,a:A):B=I(0)*a+E(0,a);A=E(0,16);C=B[:,None]*b+A[None,:];F=(B[:,None]<c)&(A[None,:]<b);G=D(k+C,F,0);J=D(l+C,F,0);M(i+I(0)*16+A,H(G*J,0))
@triton.jit
def V(r,h,s,n,m,k,j,c:A,b:A,a:A,d:A,e:A):
	F,A=E(0,e),E(0,16);G=H(D(s+F[:,None]*16+A[None,:],F[:,None]<d,0),0);L=D(n+A,A<b,1);N=G/L;J=I(0)*a+E(0,a);B=J[:,None]*b+A[None,:];C=(J[:,None]<c)&(A[None,:]<b);O,P=D(r+B,C,0),D(h+B,C,0);K=O+P*N[None,:];M(k+B,K,C);M(j+B,K,C)
	if I(0)==0:M(m+A,G,A<b)
@triton.jit
def J(k,i,c:A,b:A,a:A):B=I(0)*a+E(0,a);A,F=B//b,B%b;G=D(k+B,A<c,other=0);H=G.to(N);J=(G-H.to(C)).to(N);M(i+A*(2*b)+F,H,A<c);M(i+A*(2*b)+F+b,J,A<c)
import gc,math,torch,torch.nn.functional as F
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False
s,AE,AF=.02,.1,.6
AG,AH=.2,512
A2,AI,AJ,AK,AL=70,8.,2.,.15,5
A7,=3.,
v={}
def AM(a):return c.cholesky_ex(a)[0]
def A8(a):return a*(a.shape[1]**.5/a.norm(dim=1,keepdim=True).clamp_min(1e-06))**AF
def AN(d):A=d.square().sum(1,keepdim=True);B=O([d,A,torch.ones_like(A)],1);C=O([2*s*d,torch.full_like(A,-s),.0-s*A],1).T.contiguous();return(B@C).exp_()
def AO(c):B=c.square().sum(1,keepdim=True);A=G((len(c),64),device=c.device,dtype=f);_k[L(len(c),32),](c,B,A,len(c),c.shape[1],32,num_warps=4);return A,A
def AP(b,c):A=b.min(1,keepdim=True).values;return torch.softmax(-(b-A)/(c*A).clamp_min(1e-12),1)
def AQ(k,l):D,E=k.shape;G=l.sum(0);A=(l.T@k*(G*D).rsqrt()[:,None]).T.double();C=1.-1./D+.001;F=A0(E,device=k.device,dtype=u);H=F[:10,:10]-A.T@A/C;I=AM(H);J=torch.cholesky_solve(A.T,I);B=F/C+3./C**2*(A@J);B=(B*(E/B.trace())).float();return B
def A9(train_x,train_y,test_x):
	D=train_y;I=train_x;E=math.exp(A7);W,X=AE*E,.0001*E
	with torch.no_grad():G,Y=A8(I.float()),A8(test_x.float());H=len(G);J=F.one_hot(D.long(),10).float();K=AQ(I,J);K=AZ(I,G,J,D,K);A=O([G,Y])@K;P=A[:H].square().sum(1).mean();Q=A.norm(dim=1,keepdim=True).clamp_min(1e-06);L=P.sqrt();A=O([A*(L/Q),.5*L*(Q/L).log()],1);A=A*(P/A[:H].square().sum(1).mean()).sqrt();M=Ac(A,D,J,E,W,X);R=M.topk(2,1).values;S=R[:,0]-R[:,1];T=S>=AG;N=M.argmax(1);B=S.topk(AH,largest=False).indices;C=M[B]/E;U=C.topk(AL,1).indices;Z=O([D.long(),N]);a=O([p(H,dtype=A1,device=G.device),T]);b=AP(AC(A[H:][B],A,Z,a,U),AJ);C=C+AK*q(C).scatter_(1,U,b);V=N.clone();V[B]=torch.where(T[B],N[B],C.argmax(1));return V
def rfm_d3(train_x,train_y,test_x):
	F=test_x;G=train_y;H=train_x;C=0
	if C not in v:
		A=[H.clone(),G.clone(),F.clone()];D=d.Stream();D.wait_stream(d.current_stream())
		with d.stream(D):
			for I in range(2):A9(*A)
		d.current_stream().wait_stream(D);B=d.CUDAGraph()
		with d.graph(B):E=A9(*A)
		v[C]=B,A,E;gc.collect();gc.freeze()
		for J in range(64):B.replay()
		d.synchronize()
	B,A,E=v[C];A[0].copy_(H);A[1].copy_(G);A[2].copy_(F);B.replay();return E.clone()
AA,y=128,4
@triton.jit
def g(j,F,v,g:A,c:A,f:A,a:A,d:A,K:A,J:A,i:A,e:A,b:A,h:A):
	B=I(0)*K+E(0,K);H=E(0,64);S=E(0,J);L=E(0,16);P:A=1;T=D(j+(c+B[:,None])*f+H[None],(B[:,None]<g)&(H[None]<a),0);U=D(j+(c+B)*f+a,B<g,0).to(C)+D(j+(c+B)*f+P+a,B<g,0).to(C);O=o((K,16),0,C);R=w(c,J*i)
	for V in range(R):G=(I(1)*R+V)*J+S;W=D(j+G[None]*f+H[:,None],(H[:,None]<a)&(G[None]<c),0);X=Q(T,W);Y=D(j+G*f+a,G<c,0).to(C)+D(j+G*f+P+a,G<c,0).to(C);Z=(h+2*b*X-b*(U[:,None]+Y[None])).to(N).to(C);k=x(Z).to(N);l=D(F+G[:,None]*d+L[None],(G[:,None]<c)&(L[None]<d),0);m=Q(k,l);O=tl.inline_asm_elementwise('add.rn.f32 $0, $1, $2;',constraints='=f,f,f',args=[O,m],dtype=C,is_pure=True,pack=1)
	M(v+(I(1)*g+B[:,None])*d+L[None],O,(B[:,None]<g)&(L[None]<d))
def AR(f,h,i,j):B=j.abs().amax(0,keepdim=True).clamp_min(1e-30);D=(j/B).half();A=len(f)-i;C=G((y,A,j.shape[1]),device=j.device);g[L(A,AA),y](f,D,C,A,i,f.shape[1],AD,j.shape[1],AA,64,y,False,s,A7,num_warps=4);E=C.sum(0)*B;return E
def A3(p,o,q,r=None):
	C=o[0];B,A=q.shape
	if r is None:r=G((B,2*A),device=q.device,dtype=f);J[L(B*A,1024),](q,r,B,A,1024,num_warps=4)
	E=A5(p.T,r,out_dtype=t);I=C@(E[:,:A].double()+E[:,A:].double());F=G((len(C),2*A),device=q.device,dtype=f);J[L(len(C)*A,1024),](I,F,len(C),A,1024,num_warps=4);M=A5(p,F,out_dtype=t);H=Y(q);D=G((L(B,256),16),device=q.device);K(m,(len(D),),4,M,q,H,D,o[1],B,A,256);return H,D
def AS(g,k,j,i,b):
	S=j;N=b;O=i;E,F=k.shape;B=L(E,256);P=G((B,16),device=k.device);T,J=q(k),k.clone();A,Z=A3(O,N,J);C,H=Z.sum(0),A.half();Q=G((E,2*F),device=k.device,dtype=f)
	for U in range(S):
		W=g.mv(H,U%2==1);e[B,](A,W,P,E,F,256,num_warps=4);K(h,(B,),4,T,J,A,W,P,C,Q,E,F,256,B,X(B))
		if U+1==S:break
		R,D=A3(O,N,J,Q);I,H=Y(A),Y(A,dtype=f);M=Y(C);K(V,(B,),4,R,A,D,C,M,I,H,E,F,256,len(D),X(len(D)));A,C=I,M
	R,D=A3(O,N,J,Q);I,H=Y(A),Y(A,dtype=f);M=Y(C);K(V,(B,),4,R,A,D,C,M,I,H,E,F,256,len(D),X(len(D)));return T+I*(C/P.sum(0))
@triton.jit
def h(x,t,j,h,r,p,u,c:A,b:A,a:A,d:A,e:A):K,B=E(0,e),E(0,16);P=H(D(r+K[:,None]*16+B[None],K[:,None]<d,0),0);L=D(p+B,B<b,0)/P;J=I(0)*a+E(0,a);F=J[:,None]*b+B[None];A=(J[:,None]<c)&(B[None]<b);Q,G=D(x+F,A,0),D(t+F,A,0);R,S=D(j+F,A,0),D(h+F,A,0);G=G-S*L[None];M(x+F,Q+R*L[None],A);M(t+F,G,A);O=G.to(N);T=(G-O.to(C)).to(N);M(u+J[:,None]*(2*b)+B[None],O,A);M(u+J[:,None]*(2*b)+B[None]+b,T,A)
def AT(b):A=Y(b);K(B,(1,),4,b,A,len(b),X(len(b)),7);return A
@triton.jit
def T(f,a,e,b:A,c:A,d:A,g:A):B=I(0);J,K=B//c,B%c;C=E(0,g);A=E(0,64);L=D(f+((J*d+C[:,None])*c+K)*64+A[None,:],C[:,None]<d,0);F=H(L,0);G=H(P(A==b,F,0),0);N=D(a+B*b+A,A<b,0);O=P(G>0,F/U(G,1),N);M(e+B*b+A,O,A<b)
def AU(v,s,u=32):
	A=u;D=1400;F,M=z(s,p(len(v),device=v.device,dtype=A1),D);H=O([v,q(v[:1])])[F].contiguous();J=F<len(v);B=H[:,:A].reshape(10*A,-1).contiguous();C=L(D,128);E=G((10,C,A,64),device=v.device,dtype=t)
	for N in range(2):I=Y(B);K(W,(10,C),4,H,J,B,E,E,D,v.shape[1],A,X(A),C,128,False,True);K(T,(10*A,),4,E,B,I,v.shape[1],A,C,X(C));B=I
	return B
@triton.jit
def l(i,s,k,c:A,b:A,a:A):
	C=I(0);A=E(0,a);J=D(i+C*c+A,A<c,0);F=J.to(tl.uint16,bitcast=True).to(n)&32767;B=o((),0,n)
	for Q in range(15):K=B|1<<14-Q;R=H(((F<K)&(A<c)).to(n),0);B=P(R<b,K,B)
	L=(F<B)&(A<c);N=(F==B)&(A<c);S=b-H(L.to(n),0);G=L|N&(tl.cumsum(N.to(n),0)<=S);O=tl.cumsum(G.to(n),0)-1;M(s+C*b+O,J,G);M(k+C*b+O,A,G)
def AV(f,g=70):A=f;B,C=A.shape;D=G((B,g),device=A.device,dtype=A.dtype);E=G((B,g),device=A.device,dtype=torch.int64);l[B,](A,D,E,C,g,X(C),num_warps=8);return D,E
@triton.jit
def m(l,m,k,h,g,c:A,b:A,a:A):A=I(0)*a+E(0,a);B=E(0,16);C=(A[:,None]<c)&(B[None,:]<b);F=D(m+A[:,None]*b+B[None,:],C,0);J=D(l+A[:,None]*(2*b)+B[None,:],C,0);K=D(l+A[:,None]*(2*b)+b+B[None,:],C,0);L=D(g+A,A<c,0);G=F*L[:,None]-(J+K);M(k+A[:,None]*b+B[None,:],G,C);M(h+I(0)*16+B,H(F*G,0))
class AW:
	def __init__(A,pans,pp,n):A.pans,A.pp,A.n=pans,pp,n
	def mv(A,rhs,rev):B=rhs;D=G((32,A.pp,B.shape[1]),device=B.device);C=G((A.n,B.shape[1]),device=B.device);K(_h,(L(A.pp,128),2,16),4,A.pans,B,D,A.n,A.pp,B.shape[1],4,128,128,2,rev);K(_i,(L(C.numel(),512),),4,D,C,A.n,A.pp,B.shape[1],4,2,512);return C
def AX(a,b):return torch.bmm(a.half(),b.half(),out_dtype=t)
def AY(p,q):A=p.double();E=q[:48].T.double();C=A@E;D=A@C;F=A@D;G=1e-07*A.square().sum()/48;H=E.T@C;I=C.T@D;J=D.T@F;B=J+G*I+G.square()*(H+A0(48,device=p.device,dtype=u)*(1e-07*H.trace()/48));B=(B+B.T)*.5;K,M=c.cholesky_ex(B);L=c.solve_triangular(K,F.T,upper=False).T;return L.float()
def AZ(raw,x,y,labels,initial):D=initial;C=labels;E=torch.eye(60,device=x.device);N=A4((.5*E+.5*D)[None])[0][0];F,W=z(C,torch.ones(len(x),device=x.device,dtype=torch.bool),1400);O=torch.cat([x,torch.zeros_like(x[:1])])[F].contiguous();P=F<len(x);G=torch.empty((10,8,64,64),device=x.device);_a[10,8](O,P,G,num_warps=4,enable_fp_fusion=False);A=G.sum(1);H=A[:,60,60];B=A[:,:60,60]/H[:,None].clamp_min(1);Q=A[:,:60,:60]/H[:,None,None].clamp_min(1)-B[:,:,None]*B[:,None,:];R,S=A4(.6*Q+.4*N);I=(x[:2048][None]-B[:,None]).contiguous();J=-AX(I,R);T=(.5*(I*J).sum(2)-.5*S[:,None]).T.contiguous();K=torch.empty((256,60,60),device=x.device);L=torch.empty((2048,60),device=x.device);_j[256,](J,T,C,K,L,num_warps=4,enable_fp_fusion=False);M=K.sum(0);U=.8*AT(M)+.196*D+.004*E;V=AY(U,x);return Af(x,V,L,M)
@triton.jit
def W(h,d,J,f,F,K:A,L:A,e:A,b:A,a:A,G:A,g:A,c:A):
	U,Y=I(0),I(1);B=Y*G+E(0,G);O=E(0,64);R=E(0,b);V=D(h+(U*K+B[:,None])*L+O[None,:],(B[:,None]<K)&(O[None,:]<L),0);A=D(J+(U*e+R[:,None])*L+O[None,:],(R[:,None]<e)&(O[None,:]<L),0)
	if g:W,X=V.to(N),A.to(N);m,n=(V-W.to(C)).to(N),(A-X.to(C)).to(N);T=Q(W,S(X))+Q(m,S(X))+Q(W,S(n));T=T-.5*H(A*A,1)[None,:]
	else:o=(-.5*H(A*A,1)).to(N).to(C);T=(Q(V.to(N),S(A.to(N)))+o[None,:]).to(N).to(C)
	T=P(R[None,:]<e,T,float('-inf'));Z=tl.argmax(T,1,tie_break_left=True)
	if g:M(F+U*K+B,Z,B<K)
	if c:i=D(d+U*K+B,B<K,0);j=((Z[None,:]==R[:,None])&i[None,:]).to(N);k=P(O[None,:]==L,i[:,None].to(C),V);l=k.to(N);p=(k-l.to(C)).to(N);q=Q(j,l)+Q(j,p);M(f+((U*a+Y)*e+R[:,None])*64+O[None,:],q,R[:,None]<e)
@triton.jit
def _b(h,c,f,e:A,g:A,d:A,a:A,b:A):A=I(0)*a+E(0,a);B=I(1)*b+E(0,b);C=E(0,64);F=D(h+A[:,None]*d+C[None,:],(A[:,None]<e)&(C[None,:]<d),0);G=D(c+B[:,None]*d+C[None,:],(B[:,None]<g)&(C[None,:]<d),0);J=H(F*F,1);K=H(G*G,1);L=G*.04;N=_n(F,S(L));O=N-J[:,None]*.02-K[None,:]*.02;P=x(O);M(f+A[:,None]*g+B[None,:],P,(A[:,None]<e)&(B[None,:]<g))
@triton.jit
def _c(b,i,g,c,f:A,h:A,e:A,d:A,a:A):A=I(0);B=E(0,a);C=D(b+A*h+B,B<h,0);F=D(i+A);H=U(1.-r(C*C,0)/(1.+d),.0);G=e+F*H;J=(k(F)*C/G).to(N);M(g+A*h+B,J,B<h);M(c+A,1./G)
def Aa(a,y,A,z,x,v,w):M=x;F=v;H=w;D=a;B,Q=D.shape;C=len(H);N=G((B,C),device=D.device);K(_b,(L(B,32),L(C,64)),4,D,H,N,B,C,Q,32,64);O=AN(H).double();O.diagonal().add_(M/F);I=G((B,C),device=D.device,dtype=f);J=G(B,device=D.device);K(_c,(B,),4,N,A,I,J,B,C,z/F,M/F,X(C));E=Ae(I,J)+O;E=(E+E.T)*.5;R,T=c.cholesky_ex(E);P=c.solve_triangular(R,A0(C,device=D.device,dtype=u),upper=False);S=(P.T@P).double();return I,(S,J)
def Ab(x,w):E=w;A=x;C,F=A.shape;D=AU(A,E,32);I=G(C,device=A.device);J=G(C,device=A.device,dtype=j);K(_d,(L(C,32),),8,A,D,E,I,J,C,F);M=I.argsort(stable=True);N,H=M[:4096],M[4096:];B=L(len(H),128);P=G((10,B,32,64),device=A.device);K(_e,(10,B),4,A,H,J,P,len(H),F,B);Q=Y(D);R=G(320,device=A.device);K(_f,(320,),4,P,D,Q,R,F,B,X(B));S=torch.arange(10,device=A.device).repeat_interleave(32);T=O([A[N],Q]);U=O([E[N],S]);V=O([p(4096,device=A.device),R]);return T,U,V,D
def Ac(c,C,b,A,a,B):
	J=a;S=A;K=C;L=c;M=len(K);E,N,D,T=Ab(L[:M],K);G=len(E);P=F.one_hot(N.long(),10).float()*D.sqrt()[:,None];U=O([E,L[M:]]);Q=d.current_stream();H=d.Stream();H.wait_stream(Q)
	with d.stream(H):V,W=Aa(E,N,D,J,B,S,T)
	R,X=AO(U);Y,Z=Ad(R,G,D,J);Q.wait_stream(H);I=AS(AW(Y,Z,G),O([P,P[:,:6]],1),5,V,W);I=(I*D.sqrt()[:,None]).contiguous();return AR(R,X,G,I)[:,:10]
@triton.jit
def _d(g,b,d,f,a,e:A,c:A):A=I(0)*32+E(0,32);B=E(0,512);C=E(0,64);F=D(g+A[:,None]*c+C[None,:],(A[:,None]<e)&(C[None,:]<c),0);G=D(b+B[:,None]*c+C[None,:],(B[:,None]<320)&(C[None,:]<c),0);L=_n(F,S(G))-.5*H(G*G,1)[None,:];Q=D(d+A,A<e,-1);N=(B[None,:]//32==Q[:,None])&(B[None,:]<320);J=P(N,L,float('-inf'));R=tl.argmax(J,1,tie_break_left=True);O=r(P(~N&(B[None,:]<320),L,float('-inf')),1);K=r(J,1)-O;K=K/U(H(F*F,1)-r(J,1)-O,1e-06);M(f+A,K,A<e);M(a+A,R,A<e)
@triton.jit
def _e(g,c,a,f,d:A,b:A,e:A):G=I(0);H=I(1);A=H*128+E(0,128);B=E(0,64);J=E(0,32);K=D(c+A,A<d,0);R=D(a+K,A<d,-1);F=D(g+K[:,None]*b+B[None,:],(A[:,None]<d)&(B[None,:]<b),0);F=P(B[None,:]==b,1.,F);L=((R[None,:]==G*32+J[:,None])&(A[None,:]<d)).to(N);O=F.to(N);S=(F-O.to(C)).to(N);T=Q(L,O)+Q(L,S);M(f+((G*e+H)*32+J[:,None])*64+B[None,:],T)
@triton.jit
def _f(f,b,e,g,c:A,d:A,a:A):B=I(0);J=B//32;K=B%32;F=E(0,a);A=E(0,64);L=D(f+((J*d+F[:,None])*32+K)*64+A[None,:],F[:,None]<d,0);G=H(L,0);C=H(P(A==c,G,.0),0);N=D(b+B*c+A,A<c,0);O=P(C>0,G/U(C,1.),N);M(e+B*c+A,O,A<c);M(g+B,C)
@triton.jit
def _g(d,j,h,i:A,g:A,e:A,f:A):K=I(2);L=((k((8*K+1).to(C))-1)*.5).to(n);O=K-L*(L+1)//2;A=I(0)*e+E(0,e);B=I(1)*f+E(0,f);G=O*i+A;H=L*i+B;J=E(0,64);R=D(d+G[:,None]*64+J[None,:],(A[:,None]<i)&(J[None,:]<49),0);S=D(d+H[None,:]*64+J[:,None],(B[None,:]<i)&(J[:,None]<49),0);T=D(d+G*64+49,A<i,0).to(C)+D(d+G*64+50,A<i,0).to(C);U=D(d+H*64+49,B<i,0).to(C)+D(d+H*64+50,B<i,0).to(C);V=Q(R,S);W=(3.+.04*V-.02*(T[:,None]+U[None,:])).to(N).to(C);F=x(W).to(N).to(C);X=D(j+G,A<i,0);Y=D(j+H,B<i,0);F=F*k(X[:,None]*Y[None,:]);F=P(G[:,None]==H[None,:],F+g,F).to(N);M(h+(K*i+A[:,None])*i+B[None,:],F,(A[:,None]<i)&(B[None,:]<i))
@triton.jit
def _h(i,k,h,f:A,j:A,e:A,g:A,d:A,c:A,m:A,l:A):
	H=I(2);J=H//g;F=H%g;N=U(J,F);S=tl.minimum(J,F);O=N*(N+1)//2+S;A=I(0)*d+E(0,d);T=E(0,c);G=E(0,16);K=w(j,c*m);L=o((d,16),.0,C)
	for P in range(K):
		V=K-1-P if l else P;B=(I(1)*K+V)*c+T
		if J>F:R=D(i+(O*j+B[None,:])*j+A[:,None],(A[:,None]<j)&(B[None,:]<j),0)
		else:R=D(i+(O*j+A[:,None])*j+B[None,:],(A[:,None]<j)&(B[None,:]<j),0)
		W=D(k+(F*j+B[:,None])*e+G[None,:],(B[:,None]<j)&(G[None,:]<e),0);L=Q(R,W,L)
	M(h+((H*m+I(1))*j+A[:,None])*e+G[None,:],L,(A[:,None]<j)&(G[None,:]<e))
@triton.jit
def _i(f,e,c:A,g:A,b:A,d:A,h:A,a:A):
	B=I(0)*a+E(0,a);A=B//b;G=B%b;H=A//g;J=A%g;F=o((a,),.0,C)
	for K in range(d):
		for L in range(h):F+=D(f+(((H*d+K)*h+L)*g+J)*b+G,A<c,0)
	M(e+B,F,A<c)
def Ad(c,h,g,e):A=h//4;assert A*4==h;B=G((10,A,A),device=c.device,dtype=f);K(_g,(L(A,128),L(A,128),len(B)),8,c,g,B,A,e,128,128);return B,A
def K(b,a,c,*A):return b[a](*A,num_warps=c,enable_fp_fusion=False)
@triton.jit
def _k(f,c,d,e:A,b:A,a:A):A=I(0)*a+E(0,a);B=E(0,64);H=D(f+A[:,None]*b+B[None,:],(A[:,None]<e)&(B[None,:]<b),0).to(N);F=D(c+A,A<e,0);G=F.to(N);J=(F-G.to(C)).to(N);K=P(B[None,:]==b,G[:,None],P(B[None,:]==b+1,J[:,None],H));M(d+A[:,None]*64+B[None,:],K,A[:,None]<e)
@triton.jit
def _l(a,e,d,c:A,b:A,f:A,g:A,h:A,_natural:A,_ntrain:A):
	Q=_ntrain;J=I(0);B=E(0,b);F=E(0,b);K=(B[:,None]<c)&(F[None,:]<c);A=D(a+J*f+B[:,None]*g+F[None,:]*h,K,0)
	if _natural:
		if(J==0)&(Q>0):A=A/Q;R=.0001
		else:R=.1
		S=((B[:,None]==F[None,:])&K).to(C);A=A+S*(R*H(H(A*S,0),0)/60)
	T=o((),.0,A6)
	for G in range(c):L=H(P(F[None,:]==G,A,.0),1);N=H(P(B==G,L,.0),0);O=L/N;A=A-L[:,None]*O[None,:];A=P(B[:,None]==G,O[None,:],A);A=P(F[None,:]==G,O[:,None],A);A=P((B[:,None]==G)&(F[None,:]==G),-1./N,A);T+=tl.log(N.to(C)).to(A6)
	M(e+J*c*c+B[:,None]*c+F[None,:],-A,K);M(d+J,T)
def A4(a,_natural=False,_ntrain=1):A=G(a.shape,device=a.device);B=G(len(a),device=a.device);K(_l,(len(a),),4,a,A,B,60,64,*a.stride(),_natural,_ntrain);return A,B
@triton.jit
def _m(g,b,c,d:A,a:A,e:A,f:A):
	B=I(0)*f+E(0,f);F=I(1)*f+E(0,f);K=I(2);R=E(0,64);H=o((f,f),.0,C);J=o((f,f),.0,C);L=w(d,64*e)
	for T in range(L):A=(K*L+T)*64+R;O=D(g+A[:,None]*a+B[None,:],(A[:,None]<d)&(B[None,:]<a),0);G=D(g+A[:,None]*a+F[None,:],(A[:,None]<d)&(F[None,:]<a),0).to(C);U=D(b+A,A<d,1.);G=G/U[:,None];P=G.to(N);V=(G-P.to(C)).to(N);H=Q(S(O),P,H);J=Q(S(O),V,J)
	M(c+(K*a+B[:,None])*a+F[None,:],H+J,(B[:,None]<a)&(F[None,:]<a))
def Ae(a,b):C,A=a.shape;B=G((16,A,A),device=a.device);_m[5,5,16](a,b,B,C,A,16,64,num_warps=4,num_stages=1,enable_fp_fusion=False);return B.double().sum(0)
@triton.jit
def _a(DATA,VALID,OUT):
	J=I(0);N=I(1);A=E(0,64);U=E(0,128);B=tl.full((64,64),.0,C);K=B;L=B
	for V in range(2):F=N*256+V*128+U;O=D(VALID+J*1400+F,F<1400,0);G=D(DATA+(J*1400+F[:,None])*60+A[None,:],(F[:,None]<1400)&(A[None,:]<60),0);G=P(A[None,:]==60,1.,G);H=G.to(tl.float16);R=(G-H.to(C)).to(tl.float16);T=P(O[:,None],H,.0);W=P(O[:,None],R,.0);B=Q(S(T),H,B);K=Q(S(T),R,K);L=Q(S(W),H,L)
	M(OUT+((J*8+N)*64+A[:,None])*64+A[None,:],B+K+L)
@triton.jit
def _j(GRAD,LOGIT,LABEL,OUT,DIFF):F=I(0)*8+E(0,8);B=E(0,16);A=E(0,64);J=D(LOGIT+F[:,None]*10+B[None,:],B[None,:]<10,float('-inf'))*.25;C=tl.exp(J-tl.max(J,1)[:,None]);C=C/H(C,1)[:,None];G=D(GRAD+(B[None,:,None]*2048+F[:,None,None])*60+A[None,None,:],(B[None,:,None]<10)&(A[None,None,:]<60),.0);K=H(G*C[:,:,None],1);N=D(LABEL+F);O=H(P(B[None,:,None]==N[:,None,None],G,.0),1);M(DIFF+F[:,None]*60+A[None,:],.25*(K-O),A[None,:]<60);L=tl.reshape((G-K[:,None,:])*k(C)[:,:,None],(128,64)).to(tl.float16);R=Q(S(L),L);M(OUT+(I(0)*60+A[:,None])*60+A[None,:],R,(A[:,None]<60)&(A[None,:]<60))
@triton.jit
def _o(C,G,J,INV,OUT,J0:A,J1:A):
	A=E(0,64);B=(A[:,None]<60)&(A[None,:]<60);F=(A[:,None]<60)&(A[None,:]<48);H=D(J+A[:,None]*J0+A[None,:]*J1,F,0);O=D(INV+A[:,None]*60+A[None,:],B,0);K=_n(O,H)
	if I(0)==0:P=D(G+A[:,None]*60+A[None,:],B,0);L=_n(P,K)
	else:Q=_n(S(H),K);R=D(C+A[:,None]*60+A[None,:],B,0);N=_n(R,H);M(OUT+5760+A[:,None]*48+A[None,:],N,F);L=_n(N,Q)
	M(OUT+I(0)*2880+A[:,None]*48+A[None,:],L,F)
@triton.jit
def _p(C,J,DIR,OUT,J0:A,J1:A):I=DIR;A=E(0,64);Q=(A[:,None]<60)&(A[None,:]<60);B=(A[:,None]<60)&(A[None,:]<48);F=A[:,None]*48+A[None,:];O=D(I+F,B,0);P=D(I+2880+F,B,0);K=D(I+5760+F,B,0);R=H(H(K*O,0),0)/U(H(H(K*P,0),0),1e-30);L=O-P*R;G=D(J+A[:,None]*J0+A[None,:]*J1,B,0);S=H(H(G*K,0),0);T=.1*k(H(H(G*G,0),0)/U(H(H(L*L,0),0),1e-30));N=G-T*L;V=D(C+A[:,None]*60+A[None,:],Q,0);W=H(H(N*_n(V,N),0),0);M(OUT+F,N*k(S/W),B)
def Af(x,j,diff,fisher):A=diff;B=x.T@x;E=x[:len(A)].T@A/len(A);F=A4(fisher[None],True,0)[0][0];C=G((3,60,48),device=x.device);D=G((60,48),device=x.device);K(_o,(2,),4,B,E,j,F,C,*j.stride());K(_p,(1,),4,B,j,C,D,*j.stride());return D
@triton.jit
def _n(x,y):return Q(x,y,input_precision='tf32x3')

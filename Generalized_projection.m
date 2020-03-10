%% Main m file for Generalized geometric projection Dec,2018
%step 1) Problem set-up
close all
clear
volfrac=.4;
nelx=80;
nely=40;
settings='MNA'; %GGP  %MMC  %MNA  %GP
BC='L-shape';   %L-shape  %Short_Cantiliever  %MBB
stopping_criteria='change';
switch settings
    case 'GGP'
        p.method='MMC';%MMC %MNA %GP
        q=2;%q=1
        p.zp=1 ;% parameter for p-norm/mean regularization
        p.alp=1; %parameter for MMC
        p.epsi=0.866;% parameter for MMC
        p.bet=1e-3; %parameter for MMC
        p.deltamin=1e-6; %parameter for GP
        p.r=.5;%parameter for GP
        minh=1;
        p.sigma=1;%parameter for MNA
        p.gammav=1;%parameter for GP
        p.gammac=3;%parameter for GP
        p.penalty=3;%parameter for MNA
        p.aggregation='KSl'; %parameter for the aggregation function to be used
        % IE= Induced Exponential % KS= KS function %KSl= lowerbound KS function
        % p-norm %p-mean
        p.ka=10; % parameter for the aggregation constant
        p.saturation=true; % switch for saturation
        ncx=1; % number of components in the x direction
        ncy=1; % number of components in the y direction
        Ngp=2; % number of Gauss point per sampling window
        R=0.25; % radius of the sampling window (infty norm)
        initial_d=0.5;
    case 'MMC'
        p.method='MMC';%MMC %MNA %GP
        q=2;%q=1
        p.zp=1 ;% parameter for p-norm/mean regularization
        p.alp=1; %parameter for MMC
        p.epsi=0.7;% parameter for MMC
        p.bet=1e-3; %parameter for MMC
        p.deltamin=1e-6; %parameter for GP
        p.r=1.5;%parameter for GP
        minh=1;
        p.sigma=1.5;%parameter for MNA
        p.gammav=1;%parameter for GP
        p.gammac=3;%parameter for GP
        p.penalty=3;%parameter for MNA
        p.aggregation='KS'; %parameter for the aggregation function to be used
        % IE= Induced Exponential % KS= KS function %KSl= lowerbound KS function
        % p-norm %p-mean
        p.ka=4; % parameter for the aggregation constant
        p.saturation=false; % switch for saturation
        ncx=1; % number of components in the x direction
        ncy=1; % number of components in the y direction
        Ngp=2; % number of Gauss point per sampling window
        R=sqrt(3)/2; % radius of the sampling window (infty norm)
        initial_d=1; % component initial mass
    case 'MNA'
        p.method='MNA';%MMC%MNA %GP
        q=1;%q=1
        p.zp=1 ;% parameter for p-norm/mean regularization
        p.alp=1; %parameter for MMC
        p.epsi=0.7;% parameter for MMC
        p.bet=1e-3; %parameter for MMC
        p.deltamin=1e-6; %parameter for GP
        p.r=3;%parameter for GP
        minh=1;
        p.sigma=2;%parameter for MNA
        p.gammav=1;%parameter for GP
        p.gammac=3;%parameter for GP
        p.penalty=3;%parameter for MNA
        p.aggregation='KSl'; %parameter for the aggregation function to be used
        % IE= Induced Exponential % KS= KS function %KSl= lowerbound KS function
        % p-norm %p-mean
        p.ka=10; % parameter for the aggregation constant
        p.saturation=false; % switch for saturation
        ncx=1; % number of components in the x direction
        ncy=1; % number of components in the y direction
        Ngp=1; % number of Gauss point per sampling window
        R=sqrt(1)/2; % radius of the sampling window (infty norm)
        initial_d=0.5;
    case 'GP'
        p.method='GP';%MMC%MNA %GP
        q=1;%q=1
        p.zp=1 ;% parameter for p-norm/mean regularization
        p.alp=1; %parameter for MMC
        p.epsi=0.7;% parameter for MMC
        p.bet=1e-3; %parameter for MMC
        p.deltamin=1e-6; %parameter for GP
        p.r=1.5;%parameter for GP
        minh=1;
        p.sigma=1.5;%parameter for MNA
        p.gammav=1;%parameter for GP
        p.gammac=3;%parameter for GP
        p.penalty=3;%parameter for MNA
        p.aggregation='KSl'; %parameter for the aggregation function to be used
        % IE= Induced Exponential % KS= KS function %KSl= lowerbound KS function
        % p-norm %p-mean
        p.ka=10; % parameter for the aggregation constant
        p.saturation=false; % switch for saturation
        ncx=1; % number of components in the x direction
        ncy=1; % number of components in the y direction
        Ngp=2; % number of Gauss point per sampling window
        R=sqrt(1)/2; % radius of the sampling window (infty norm)
        initial_d=0.5;  
end
cross_starting_guess=true;
rs=replace(num2str(R,'%3.2f'),'.','_');
folder_name=['Optimization_history_',BC,settings,p.method,'nelx_',num2str(nelx),'nely_',num2str(nely),'_R_',rs,'_Ngp_',num2str(Ngp),'_SC_',stopping_criteria];
%mkdir(folder_name)
Path=[folder_name,'/'];

%% MATERIAL PROPERTIES
p.E0 = 1;
p.Emin = 1e-6;
nu = 0.3;

%% PREPARE FINITE ELEMENT ANALYSIS
A11 = [12  3 -6 -3;  3 12  3  0; -6  3 12 -3; -3  0 -3 12];
A12 = [-6 -3  0  3; -3 -6 -3 -6;  0 -3 -6  3;  3 -6  3 -6];
B11 = [-4  3 -2  9;  3 -4 -9  4; -2 -9 -4 -3;  9  4 -3 -4];
B12 = [ 2 -3  4 -9; -3  2  9 -2;  4  9  2  3; -9 -2  3  2];
KE = 1/(1-nu^2)/24*([A11 A12;A12' A11]+nu*[B11 B12;B12' B11]);
nodenrs = reshape(1:(1+nelx)*(1+nely),1+nely,1+nelx);
edofVec = reshape(2*nodenrs(1:end-1,1:end-1)+1,nelx*nely,1);      % didn't translate
edofMat = repmat(edofVec,1,8)+repmat([0 1 2*nely+[2 3 0 1] -2 -1],nelx*nely,1);
iK = reshape(kron(edofMat,ones(8,1))',64*nelx*nely,1);
jK = reshape(kron(edofMat,ones(1,8))',64*nelx*nely,1);
U = zeros(2*(nely+1)*(nelx+1),1);

%define the nodal coordinates
[Yy,Xx]=find(nodenrs);
Yy=nely+1-Yy;
Xx=Xx-1;

% Element connectivity
enodeMat=edofMat(:,[2,4,6,8])/2;
% compute the centroid coordinates
xc=mean(Xx(enodeMat'));
yc=mean(Yy(enodeMat'));
centroid_coordinate=[xc(:),yc(:)];
a=-R;
b=R;
[gpc,wc]=lgwt(Ngp,a,b);
[gpcx,gpcy]=meshgrid(gpc,gpc);
gauss_weight=wc*wc';
gpcx=reshape((repmat(gpcx(:),1,size(centroid_coordinate,1)))',[],1);
gpcy=reshape((repmat(gpcy(:),1,size(centroid_coordinate,1)))',[],1);
gauss_weight=reshape((repmat(gauss_weight(:),1,size(centroid_coordinate,1)))',[],1);
cc=repmat(centroid_coordinate,Ngp^2,1);
gauss_point=cc+[gpcx,gpcy];
[ugp,~,idgp]=unique(gauss_point,'rows');

%% DEFINE LOADS AND SUPPORTS
switch BC
    case 'MBB'
        excitation_node=1;
        excitation_direction=2;
        amplitude=-1;
        F = sparse(2*(excitation_node-1)+excitation_direction,1,amplitude,2*(nely+1)*(nelx+1),1);
        fixednodes=[find(Xx==min(Xx));(nelx+1)*(nely+1)];
        fixed_dir=[ones(nely+1,1);2];
        fixeddofs=2*(fixednodes-1)+fixed_dir;
        emptyelts=[]; 
        fullelts = [];
    case 'Short_Cantiliever'
        excitation_node=find((Xx==max(Xx))&(Yy==fix(0.5*min(Yy)+0.5*max(Yy))));     % Number of the node where the force is applied
        excitation_direction=2;         %Axis in which the force is applied
        amplitude=-1;                   % Amplitude of the force
        F = sparse(2*(excitation_node-1)+excitation_direction,1,amplitude,2*(nely+1)*(nelx+1),1); %Force matrix
        fixednodes=repmat(find(Xx==min(Xx)),2,1);   % Fixed Nodes (1 to 31)
        fixed_dir=[ones(nely+1,1);2*ones(nely+1,1)]; % Directions fixed on the fixed nodes
        fixeddofs=2*(fixednodes-1)+fixed_dir(:);
        emptyelts=[]; 
        fullelts = [];
    case 'L-shape'
        excitation_node=find((Xx==max(Xx))&(Yy==fix(0.5*min(Yy)+0.5*max(Yy))));
        excitation_direction=2;
        amplitude=-1;
        F = sparse(2*(excitation_node-1)+excitation_direction,1,amplitude,2*(nely+1)*(nelx+1),1);
        fixednodes=repmat(find(Yy==max(Yy)),2,1);
        fixed_dir=[ones(nelx+1,1),2*ones(nelx+1,1)];
        fixeddofs=2*(fixednodes-1)+fixed_dir(:);
        emptyelts=find(xc>=(((max(Xx)+min(Xx))/2))&(yc>=((max(Yy)+min(Yy))/2)));
        fullelts = [];
    otherwise
        error('BC string should be a valid entry: ''MBB'',''L-Shape'',''Short_Cantiliever''')
end
alldofs = [1:2*(nely+1)*(nelx+1)];
freedofs = setdiff(alldofs,fixeddofs);
%% INITIALIZE ITERATION
% define the initial guess components
xp=linspace(min(Xx),max(Xx),ncx+2);
% xp=xp(2:1:end-1);
yp=linspace(min(Yy),max(Yy),ncy+2);
% yp=yp(2:1:end-1);
[xx,yy]=meshgrid(xp,yp);
if cross_starting_guess
    Xc=repmat(xx(:),2,1);
    Yc=repmat(yy(:),2,1);
    Lc=2*sqrt((nelx/(ncx+2))^2+(nely/(ncy+2))^2)*ones(size(Xc));
    Tc=atan2(nely/ncy,nelx/ncx)*[ones(length(Xc)/2,1);-ones(length(Xc)/2,1)];
    hc=2*ones(length(Xc),1);
else
    Xc=repmat(xx(:),1,1);
    Yc=repmat(yy(:),1,1);
    initial_Lh_ratio=nelx/nely;
    hc=repmat(sqrt(1/initial_Lh_ratio*(nelx*nely)/ncx/ncy),length(Xc),1);
    Lc=initial_Lh_ratio*hc;
    Tc=[0*pi/4*ones(ncx*ncy,1)];    %WTF 0*pi?????
end

Mc=initial_d*ones(size(Xc));
% initial guess vector
%add components on the boundaries
xb=linspace(min(Xx),max(Xx),5);
xb=xb([2,4]);
Xo=[xb(:);xb(:)];
Yo=[repmat(min(Yy),2,1);repmat(max(Yy),2,1)];
% if strcmp(BC,'L-shape')
%     Yo(end)=(min(Yy)+max(Yy))/2;
% end
ho=hc(1)*ones(4,1);
Lo=nelx/2*ones(4,1);
To=zeros(4,1);
Mo=initial_d*ones(4,1);
yb=linspace(min(Yy),max(Yy),5);
yb=yb([2,4]);
Xv=[repmat(min(Xx),2,1);repmat(max(Xx),2,1)];
% if strcmp(BC,'L-shape')
%     Xv(end)=(min(Xx)+max(Xx))/2;
% end
Yv=[yb(:);yb(:)];
hv=hc(1)*ones(4,1);
Lv=nely/2*ones(4,1);
Tv=pi/2*ones(4,1);
Mv=initial_d*ones(4,1);
% uncomment to add vertical and horizontal components on the boundaries
% Xc=[Xc;Xo;Xv];
% Yc=[Yc;Yo;Yv];
% Lc=[Lc;Lo;Lv];
% hc=[hc;ho;hv];
% Tc=[Tc;To;Tv];
% Mc=[Mc;Mo;Mv];
Xg=reshape([Xc,Yc,Lc,hc,Tc,Mc]',[],1);
Xl=min(Xx-1)*ones(size(Xc));
Xu=max(Xx+1)*ones(size(Xc));
Yl=min(Yy-1)*ones(size(Xc));
Yu=max(Yy+1)*ones(size(Xc));
Ll=0*ones(size(Xc));
Lu=sqrt(nelx^2+nely^2)*ones(size(Xc));
hl=minh*ones(size(Xc));
hu=sqrt(nelx^2+nely^2)*ones(size(Xc));
Tl=-2*pi*ones(size(Xc));
Tu=2*pi*ones(size(Xc));
Ml=0*ones(size(Xc));
Mu=ones(size(Xc));
dmin=sqrt(2)/2;
lower_bound=reshape([Xl,Yl,Ll,hl,Tl,Ml]',[],1);
upper_bound=reshape([Xu,Yu,Lu,hu,Tu,Mu]',[],1);
X=(Xg-lower_bound)./(upper_bound-lower_bound);
loop = 0;
change = 1;
m = 1;
n = length(X(:));
epsimin = 0.0000001;
eeen    = ones(n,1);
eeem    = ones(m,1);
zeron   = zeros(n,1);
zerom   = zeros(m,1);
xval    = X(:);
xold1   = xval;
xold2   = xval;
xmin    = zeron;
xmax    = eeen;
low     = xmin;
upp     = xmax;
C       = 1000*eeem;
d       = 0*eeem;
a0      = 1;
a       = zerom;
outeriter = 0;
maxoutit  = 2000;
kkttol  =0.001;
changetol=0.001;
kktnorm = kkttol+10;
outit = 0;
change=1;

%% START ITERATION
cvec=zeros(maxoutit,1);
vvec=cvec;ovvec=cvec;gvec=cvec;pvec=cvec;
plot_rate=10;
transition=50;
change_of_formultion=200;
change_of_formultion2=20000;
active_elements=setdiff(1:nelx*nely,[emptyelts(:);fullelts(:)]);
%initialize variables for plot
tt=0:0.005:(2*pi);tt=repmat(tt,length(Xc),1);
cc=cos(tt);ss=sin(tt);

switch stopping_criteria
    case 'kktnorm'
        stop_cond=outit < maxoutit && kktnorm>kkttol;
    case 'change'
        stop_cond=outit < maxoutit && change>changetol;
end

while  stop_cond
    %     change>0.001&&
    outit   = outit+1;
    outeriter = outeriter+1;    

    %% Project component on DZ
    [W,dW_dX,dW_dY,dW_dT,dW_dL,dW_dh]=Wgp(ugp(:,1),ugp(:,2),Xg,p);

    %generalized projection
    delta=sum(  reshape(W(:,idgp).*repmat(gauss_weight(:)',size(W,1),1),size(W,1),[],Ngp^2),3)./sum(reshape(repmat(gauss_weight(:)',size(W,1),1),size(W,1),[],Ngp^2),3);        
    ddelta_dX=sum(reshape(dW_dX(:,idgp).*repmat(gauss_weight(:)',size(dW_dX,1),1),size(dW_dX,1),[],Ngp^2),3)./sum(reshape(repmat(gauss_weight(:)',size(dW_dX,1),1),size(dW_dX,1),[],Ngp^2),3);
    ddelta_dY=sum(reshape(dW_dY(:,idgp).*repmat(gauss_weight(:)',size(dW_dX,1),1),size(dW_dX,1),[],Ngp^2),3)./sum(reshape(repmat(gauss_weight(:)',size(dW_dX,1),1),size(dW_dX,1),[],Ngp^2),3);
    ddelta_dT=sum(reshape(dW_dT(:,idgp).*repmat(gauss_weight(:)',size(dW_dX,1),1),size(dW_dX,1),[],Ngp^2),3)./sum(reshape(repmat(gauss_weight(:)',size(dW_dX,1),1),size(dW_dX,1),[],Ngp^2),3);
    ddelta_dL=sum(reshape(dW_dL(:,idgp).*repmat(gauss_weight(:)',size(dW_dX,1),1),size(dW_dX,1),[],Ngp^2),3)./sum(reshape(repmat(gauss_weight(:)',size(dW_dX,1),1),size(dW_dX,1),[],Ngp^2),3);
    ddelta_dh=sum(reshape(dW_dh(:,idgp).*repmat(gauss_weight(:)',size(dW_dX,1),1),size(dW_dX,1),[],Ngp^2),3)./sum(reshape(repmat(gauss_weight(:)',size(dW_dX,1),1),size(dW_dX,1),[],Ngp^2),3);    
    delta_c=sum(reshape(W(:,idgp).^q.*repmat(gauss_weight(:)',size(W,1),1),size(W,1),[],Ngp^2),3)./sum(reshape(repmat(gauss_weight(:)',size(W,1),1),size(W,1),[],Ngp^2),3);
    ddelta_c_dX=sum(reshape(q*dW_dX(:,idgp).*W(:,idgp).^(q-1).*repmat(gauss_weight(:)',size(dW_dX,1),1),size(dW_dX,1),[],Ngp^2),3)./sum(reshape(repmat(gauss_weight(:)',size(dW_dX,1),1),size(dW_dX,1),[],Ngp^2),3);
    ddelta_c_dY=sum(reshape(q*dW_dY(:,idgp).*W(:,idgp).^(q-1).*repmat(gauss_weight(:)',size(dW_dX,1),1),size(dW_dX,1),[],Ngp^2),3)./sum(reshape(repmat(gauss_weight(:)',size(dW_dX,1),1),size(dW_dX,1),[],Ngp^2),3);
    ddelta_c_dT=sum(reshape(q*dW_dT(:,idgp).*W(:,idgp).^(q-1).*repmat(gauss_weight(:)',size(dW_dX,1),1),size(dW_dX,1),[],Ngp^2),3)./sum(reshape(repmat(gauss_weight(:)',size(dW_dX,1),1),size(dW_dX,1),[],Ngp^2),3);
    ddelta_c_dL=sum(reshape(q*dW_dL(:,idgp).*W(:,idgp).^(q-1).*repmat(gauss_weight(:)',size(dW_dX,1),1),size(dW_dX,1),[],Ngp^2),3)./sum(reshape(repmat(gauss_weight(:)',size(dW_dX,1),1),size(dW_dX,1),[],Ngp^2),3);
    ddelta_c_dh=sum(reshape(q*dW_dh(:,idgp).*W(:,idgp).^(q-1).*repmat(gauss_weight(:)',size(dW_dX,1),1),size(dW_dX,1),[],Ngp^2),3)./sum(reshape(repmat(gauss_weight(:)',size(dW_dX,1),1),size(dW_dX,1),[],Ngp^2),3);    
 
    %model update
    [E,dE,dE_dm]=model_updateM(delta_c,p,X);
    [rho,drho_ddelta,drho_dm]=model_updateV(delta,p,X); 
    dE_dX=dE.*ddelta_c_dX;
    dE_dY=dE.*ddelta_c_dY;
    dE_dT=dE.*ddelta_c_dT;
    dE_dL=dE.*ddelta_c_dL;
    dE_dh=dE.*ddelta_c_dh;
    drho_dX=drho_ddelta.*ddelta_dX;
    drho_dY=drho_ddelta.*ddelta_dY;
    drho_dT=drho_ddelta.*ddelta_dT;
    drho_dL=drho_ddelta.*ddelta_dL;
    drho_dh=drho_ddelta.*ddelta_dh;
       

    
    
    xPhys=full(reshape(rho(:),nely,nelx));
    E=full(reshape(E(:),nely,nelx));


    
    %passive elements
    xPhys(emptyelts) = 0;
    xPhys(fullelts) = 1;
    E(emptyelts) = p.Emin;
    E(fullelts) = p.E0;
  
    
    %% FE-ANALYSIS    
    sK = reshape(KE(:)*(E(:)'),64*nelx*nely,1);
    K = sparse(iK,jK,sK);
    K = (K+K')/2;
    U(freedofs) = K(freedofs,freedofs)\F(freedofs); 
    
    %% OBJECTIVE FUNCTION AND SENSITIVITY ANALYSIS
    ce = reshape(sum((U(edofMat)*KE).*U(edofMat),2),nely,nelx);
    c = sum(sum((E).*ce));
    v=mean(xPhys(:));
    dc_dE = -ce;
    dc_dE(emptyelts) = 0;
    dc_dE(fullelts) = 0;
    dc_dX=dE_dX*dc_dE(:);
    dc_dY=dE_dY*dc_dE(:);
    dc_dL=dE_dL*dc_dE(:);
    dc_dh=dE_dh*dc_dE(:);
    dc_dT=dE_dT*dc_dE(:);
    dc_dm=dE_dm*dc_dE(:);
    dc=zeros(size(X));
    dc(1:6:end)=dc_dX;
    dc(2:6:end)=dc_dY;
    dc(3:6:end)=dc_dL;
    dc(4:6:end)=dc_dh;
    dc(5:6:end)=dc_dT;
    dc(6:6:end)=dc_dm;
    dv_dxPhys = ones(nely,nelx)/nelx/nely;
    dv_dxPhys(emptyelts) = 0;
    dv_dxPhys(fullelts) = 0;
    dv_dX=drho_dX*dv_dxPhys(:);
    dv_dY=drho_dY*dv_dxPhys(:);
    dv_dL=drho_dL*dv_dxPhys(:);
    dv_dh=drho_dh*dv_dxPhys(:);
    dv_dT=drho_dT*dv_dxPhys(:);
    dv_dm=drho_dm*dv_dxPhys(:);
    dv=zeros(size(X));
    dv(1:6:end)=dv_dX;
    dv(2:6:end)=dv_dY;
    dv(3:6:end)=dv_dL;
    dv(4:6:end)=dv_dh;
    dv(5:6:end)=dv_dT;
    dv(6:6:end)=dv_dm;
    cvec(outit)=c;vvec(outit)=v;
    f0val=log(c+1);
    fval=[(v-volfrac)/volfrac]*100;%
    df0dx=(dc(:)/(c+1).*(upper_bound(:)-lower_bound(:)));
    dfdx=[dv(:)'/volfrac]*100.*(upper_bound(:)-lower_bound(:))';
    innerit=0;
    outvector1 = [outeriter innerit f0val fval'];
    outvector2 = xval;
    c0=c;
    x0=xPhys;      
    %% PRINT RESULTS
    fprintf(' It.:%5i Obj.:%7.6e Vol.:%7.3f kktnorm.:%7.3f ch.:%7.3f\n',outit,c, ...
        mean(xPhys(:)),kktnorm,change);
    if rem(outit,plot_rate)==0
        figure(3)
        subplot(2,1,1)
        plot(1:outit,cvec(1:outit),'bo','MarkerFaceColor','b')
        % title(['Convergence  Compliance =',num2str(c,'%4.3e'),', iter = ', num2str(outit)])
        grid on
        legend(['C =',num2str(c,'%4.3e'),', iter = ', num2str(outit)],'Location','Best')
        xlabel('iter')
        axis([0 outit 0 1e3])
        subplot(2,1,2)
        plot(1:outit,vvec(1:outit)*100,'ro','MarkerFaceColor','r')
        grid on
        legend(['V = ',num2str(mean(xPhys(:))*100),', iter = ', num2str(outit)],'Location','Best')
        xlabel('iter')
        % title(['Convergence volfrac = ',num2str(mean(xPhys(:))*100),', iter = ', num2str(outit)])
        % print([Path,'convergence_',num2str(outit,'%03d')],'-dpng')
        axis([0 outit 0 Inf])
        
        %         print([Path,'convergence_',num2str(outit,'%03d')],'-dpng')
        %% PLOT DENSITIES
        figure(1)
        map=colormap(gray);
        map=map(end:-1:1,:);
        caxis([0 1])
        patchplot2 = patch('Vertices',[Xx,Yy],'Faces',edofMat(:,[2,4,6,8])/2,'FaceVertexCData',(1-xPhys(:))*[1 1 1],'FaceColor','flat','EdgeColor','none'); axis equal; axis off; ;hold on
        hold on
        fill([min(Xx),max(Xx),max(Xx),min(Xx)],[min(Yy),min(Yy),max(Yy),max(Yy)],'w','FaceAlpha',0.)
        scatter(Xx(fixednodes(fixed_dir==1)),Yy(fixednodes(fixed_dir==1)),'>b','filled')
        scatter(Xx(fixednodes(fixed_dir==2)),Yy(fixednodes(fixed_dir==2)),'^b','filled')
        scal=10;
        quiver(Xx(excitation_node),Yy(excitation_node)+scal*(excitation_direction==2),excitation_direction==1,-(excitation_direction==2),scal,'r','Linewidth',2)
        %     title('density plot')
        colormap(map)
        colorbar
        drawnow
        hold off
        axis([min(Xx),max(Xx),min(Yy),max(Yy)])
        %     print([Path,'density_',num2str(outit-1,'%03d')],'-dpng')
        %% Component Plot
        figure(2)
        Xc=Xg(1:6:end);
        Yc=Xg(2:6:end);
        Lc=Xg(3:6:end);
        hc=Xg(4:6:end);
        Tc=Xg(5:6:end);
        Mc=Xg(6:6:end);
        C0=repmat(cos(Tc),1,size(cc,2));
        S0=repmat(sin(Tc),1,size(cc,2));
        xxx=repmat(Xc(:),1,size(cc,2))+cc;
        yyy=repmat(Yc(:),1,size(cc,2))+ss;
        xi=C0.*(xxx-Xc)+S0.*(yyy-Yc);
        Eta=-S0.*(xxx-Xc)+C0.*(yyy-Yc);
        [dd]=norato_bar(xi,Eta,repmat(Lc(:),1,size(cc,2)),repmat(hc(:),1,size(cc,2)));
        xn=repmat(Xc,1,size(cc,2))+dd.*cc;
        yn=repmat(Yc,1,size(cc,2))+dd.*ss;
        %     X1=Xc+Lcsi.*cos(-Tc)-Leta.*sin(-Tc);
        %     Y1=Yc-Lcsi.*sin(-Tc)-Leta.*cos(-Tc);
        %     X2=Xc+Lcsi.*cos(-Tc)+Leta.*sin(-Tc);
        %     Y2=Yc-Lcsi.*sin(-Tc)+Leta.*cos(-Tc);
        %     X3=Xc-Lcsi.*cos(-Tc)+Leta.*sin(-Tc);
        %     Y3=Yc+Lcsi.*sin(-Tc)+Leta.*cos(-Tc);
        %     X4=Xc-Lcsi.*cos(-Tc)-Leta.*sin(-Tc);
        %     Y4=Yc+Lcsi.*sin(-Tc)-Leta.*cos(-Tc);
        tolshow=0.1;
        Shown_compo=find(Mc>tolshow);
        fill([min(Xx),max(Xx),max(Xx),min(Xx)],[min(Yy),min(Yy),max(Yy),max(Yy)],'w','FaceAlpha',0.)
        hold on
        fill(xn(Shown_compo,:)',yn(Shown_compo,:)',Mc(Shown_compo),'FaceAlpha',0.5)
        if strcmp(BC,'L-shape')
            fill([fix((min(Xx)+max(Xx))/2),max(Xx),max(Xx),fix((min(Xx)+max(Xx))/2)],[fix((min(Yy)+max(Yy))/2),fix((min(Yy)+max(Yy))/2),max(Yy),max(Yy)],'w')
        end
        %     fill([X1(Shown_compo),X2(Shown_compo),X3(Shown_compo),X4(Shown_compo)]',[Y1(Shown_compo),Y2(Shown_compo),Y3(Shown_compo),Y4(Shown_compo)]',[Mc(Shown_compo)],'FaceAlpha',0.5)
        caxis([0,1])
        colormap 'jet'
        axis equal; 
        axis off;
        hold on
%         scatter(Xx(fixednodes(fixed_dir==1)),Yy(fixednodes(fixed_dir==1)),'>b','filled')
%         scatter(Xx(fixednodes(fixed_dir==2)),Yy(fixednodes(fixed_dir==2)),'^b','filled')
        scal=10;
        quiver(Xx(excitation_node),Yy(excitation_node)+scal*(excitation_direction==2),excitation_direction==1,-(excitation_direction==2),scal,'r','Linewidth',2)
        colorbar
        %     title('component plot')
        drawnow
        axis([min(Xx),max(Xx),min(Yy),max(Yy)])
        %     print([Path,'component_',num2str(outit-1,'%03d')],'-dpng')
        hold off
    end
    
    %% MMA code optimization
    [X,ymma,zmma,lam,xsi,eta,mu,zet,S,low,upp] = ...
        mmasub(m,n,outeriter,xval,xmin,xmax,xold1,xold2, ...
        f0val,df0dx,fval,dfdx,low,upp,a0,a,C,d);    
    xold2 = xold1;
    xold1 = xval;
    xval  = X;
    Xg=lower_bound+(upper_bound-lower_bound).*X;
    change=norm(xval-xold1);    
    %% %% The residual vector of the KKT conditions is calculated:
    [residu,kktnorm,residumax] = ...
        kktcheck(m,n,X,ymma,zmma,lam,xsi,eta,mu,zet,S, ...
        xmin,xmax,df0dx,fval,dfdx,a0,a,C,d);
    outvector1 = [outeriter innerit f0val fval(:)'];
    outvector2 = xval;
    switch stopping_criteria
        case 'kktnorm'
            stop_cond=outit < maxoutit && kktnorm>kkttol;
        case 'change'
            stop_cond=outit < maxoutit &&change>changetol;
    end
end

%% PLOT DENSITIES
figure(1)
map=colormap(gray);
map=map(end:-1:1,:);
caxis([0 1])
patchplot2 = patch('Vertices',[Xx,Yy],'Faces',edofMat(:,[2,4,6,8])/2,'FaceVertexCData',(1-xPhys(:))*[1 1 1],'FaceColor','flat','EdgeColor','none'); axis equal; axis off; ;hold on
hold on
fill([min(Xx),max(Xx),max(Xx),min(Xx)],[min(Yy),min(Yy),max(Yy),max(Yy)],'w','FaceAlpha',0.)
scatter(Xx(fixednodes(fixed_dir==1)),Yy(fixednodes(fixed_dir==1)),'>b','filled')
scatter(Xx(fixednodes(fixed_dir==2)),Yy(fixednodes(fixed_dir==2)),'^b','filled')
scal=10;
quiver(Xx(excitation_node),Yy(excitation_node)+scal*(excitation_direction==2),excitation_direction==1,-(excitation_direction==2),scal,'r','Linewidth',2)
%     title('density plot')
colormap(map)
colorbar
drawnow
hold off
axis([min(Xx),max(Xx),min(Yy),max(Yy)])
FEM_enhanceDisplay
print([Path,'density_',num2str(outit,'%03d')],'-dpng')
%% Component Plot
figure(2)
Xc=Xg(1:6:end);
Yc=Xg(2:6:end);
Lc=Xg(3:6:end);
hc=Xg(4:6:end);
Tc=Xg(5:6:end);
Mc=Xg(6:6:end);
C0=repmat(cos(Tc),1,size(cc,2));S0=repmat(sin(Tc),1,size(cc,2));
xxx=repmat(Xc(:),1,size(cc,2))+cc;
yyy=repmat(Yc(:),1,size(cc,2))+ss;
xi=C0.*(xxx-Xc)+S0.*(yyy-Yc);
Eta=-S0.*(xxx-Xc)+C0.*(yyy-Yc);
[dd]=norato_bar(xi,Eta,repmat(Lc(:),1,size(cc,2)),repmat(hc(:),1,size(cc,2)));
xn=repmat(Xc,1,size(cc,2))+dd.*cc;
yn=repmat(Yc,1,size(cc,2))+dd.*ss;
%     X1=Xc+Lcsi.*cos(-Tc)-Leta.*sin(-Tc);
%     Y1=Yc-Lcsi.*sin(-Tc)-Leta.*cos(-Tc);
%     X2=Xc+Lcsi.*cos(-Tc)+Leta.*sin(-Tc);
%     Y2=Yc-Lcsi.*sin(-Tc)+Leta.*cos(-Tc);
%     X3=Xc-Lcsi.*cos(-Tc)+Leta.*sin(-Tc);
%     Y3=Yc+Lcsi.*sin(-Tc)+Leta.*cos(-Tc);
%     X4=Xc-Lcsi.*cos(-Tc)-Leta.*sin(-Tc);
%     Y4=Yc+Lcsi.*sin(-Tc)-Leta.*cos(-Tc);
tolshow=0.1;
Shown_compo=find(Mc>tolshow);
fill([min(Xx),max(Xx),max(Xx),min(Xx)],[min(Yy),min(Yy),max(Yy),max(Yy)],'w','FaceAlpha',0.)

hold on
fill(xn(Shown_compo,:)',yn(Shown_compo,:)',Mc(Shown_compo),'FaceAlpha',0.5)
if strcmp(BC,'L-shape')
    fill([fix((min(Xx)+max(Xx))/2),max(Xx),max(Xx),fix((min(Xx)+max(Xx))/2)],[fix((min(Yy)+max(Yy))/2),fix((min(Yy)+max(Yy))/2),max(Yy),max(Yy)],'w')
end
%     fill([X1(Shown_compo),X2(Shown_compo),X3(Shown_compo),X4(Shown_compo)]',[Y1(Shown_compo),Y2(Shown_compo),Y3(Shown_compo),Y4(Shown_compo)]',[Mc(Shown_compo)],'FaceAlpha',0.5)
caxis([0,1])
colormap 'jet'
axis equal; axis off;
hold on
scatter(Xx(fixednodes(fixed_dir==1)),Yy(fixednodes(fixed_dir==1)),'>b','filled')
scatter(Xx(fixednodes(fixed_dir==2)),Yy(fixednodes(fixed_dir==2)),'^b','filled')
scal=10;
quiver(Xx(excitation_node),Yy(excitation_node)+scal*(excitation_direction==2),excitation_direction==1,-(excitation_direction==2),scal,'r','Linewidth',2)
colorbar
%     title('component plot')
drawnow
axis([min(Xx),max(Xx),min(Yy),max(Yy)])
FEM_enhanceDisplay
print([Path,'component_',num2str(outit,'%03d')],'-dpng')
hold off
figure(3)
subplot(2,1,1)
plot(1:outit,cvec(1:outit),'bo','MarkerFaceColor','b')
% title(['Convergence  Compliance =',num2str(c,'%4.3e'),', iter = ', num2str(outit)])
grid on
hold on
scatter(outit,c,'k','fill')
hold off
text(outit,c,['C =',num2str(c,'%4.2f'),' at iteration ', num2str(outit)],'VerticalAlignment','bottom','HorizontalAlignment','right','FontSize',10,'FontWeight','bold')
% legend(['C =',num2str(c,'%4.3e'),', iter = ', num2str(outit)],'Location','Best')
xlabel('iter')
ylabel('C')
FEM_enhanceDisplay
% axis([0 outit 0 1e3])
subplot(2,1,2)
plot(1:outit,vvec(1:outit)*100,'ro','MarkerFaceColor','r')
grid on
hold on
scatter(outit,mean(xPhys(:))*100,'k','fill')
hold off
text(outit,mean(xPhys(:))*100,['V = ',num2str(mean(xPhys(:))*100,'%4.2f'),'% at iteration ', num2str(outit)],'VerticalAlignment','bottom','HorizontalAlignment','right','FontSize',10,'FontWeight','bold')
% legend(['V = ',num2str(mean(xPhys(:))*100),', iter = ', num2str(outit)],'Location','Best')
xlabel('iter')
ylabel('V [%]')
%%

FEM_enhanceDisplay
%% 
% axis([0 outit 0 Inf])
% title(['Convergence volfrac = ',num2str(mean(xPhys(:))*100),', iter = ', num2str(outit)])
print([Path,'convergence_',num2str(outit,'%03d')],'-dpng')

function [x,w]=lgwt(N,a,b)

% lgwt.m
%
% This script is for computing definite integrals using Legendre-Gauss 
% Quadrature. Computes the Legendre-Gauss nodes and weights  on an interval
% [a,b] with truncation order N
%
% Suppose you have a continuous function f(x) which is defined on [a,b]
% which you can evaluate at any x in [a,b]. Simply evaluate it at all of
% the values contained in the x vector to obtain a vector f. Then compute
% the definite integral using sum(f.*w);
%
% Written by Greg von Winckel - 02/25/2004
N=N-1;
N1=N+1; N2=N+2;

xu=linspace(-1,1,N1)';

% Initial guess
y=cos((2*(0:N)'+1)*pi/(2*N+2))+(0.27/N1)*sin(pi*xu*N/N2);

% Legendre-Gauss Vandermonde Matrix
L=zeros(N1,N2);

% Derivative of LGVM
Lp=zeros(N1,N2);

% Compute the zeros of the N+1 Legendre Polynomial
% using the recursion relation and the Newton-Raphson method

y0=2;

% Iterate until new points are uniformly within epsilon of old points
while max(abs(y-y0))>eps
    
    
    L(:,1)=1;
    Lp(:,1)=0;
    
    L(:,2)=y;
    Lp(:,2)=1;
    
    for k=2:N1
        L(:,k+1)=( (2*k-1)*y.*L(:,k)-(k-1)*L(:,k-1) )/k;
    end
 
    Lp=(N2)*( L(:,N1)-y.*L(:,N2) )./(1-y.^2);   
    
    y0=y;
    y=y0-L(:,N2)./Lp;
    
end

% Linear map from[-1,1] to [a,b]
x=(a*(1-y)+b*(1+y))/2;      

% Compute the weights
w=(b-a)./((1-y.^2).*Lp.^2)*(N2/N1)^2;
end

function [W,dW_dX,dW_dY,dW_dT,dW_dL,dW_dh]=Wgp(x,y,Xc,p)
%  Evaluate characteristic function in each Gauss point
ii=1:numel(x);
X=Xc(1:6:end);
Y=Xc(2:6:end);
L=Xc(3:6:end);
h=Xc(4:6:end);
T=Xc(5:6:end);
jj=1:numel(X);
[I,J]=meshgrid(ii,jj);
xi=reshape(x(I),size(I));
yi=reshape(y(I),size(I));
rho=sqrt((X(J)-xi).^2+(Y(J)-yi).^2);
drho_dX=(X(J)-xi)./(rho+(rho==0));
drho_dY=(Y(J)-yi)./(rho+(rho==0));
phi=atan2(-Y(J)+yi,-(X(J)-xi))-T(J);
dphi_dX=((-Y(J)+yi)./(rho.^2+(rho==0)));
dphi_dY=(X(J)-xi)./(rho.^2+(rho==0));
dphi_dT=-ones(size(J));
upsi=sqrt(rho.^2+L(J).^2/4-rho.*L(J).*abs(cos(phi))).*(((rho.*cos(phi)).^2)>=(L(J).^2/4))+~(((rho.*cos(phi)).^2)>=(L(J).^2/4)).*abs(rho.*sin(phi));
dupsi_drho=(2*rho-L(J).*abs(cos(phi)))/2./(upsi+(upsi==0)).*((((rho.*cos(phi)).^2)>=(L(J).^2/4)))+~(((rho.*cos(phi)).^2)>=(L(J).^2/4)).*abs(sin(phi));
dupsi_dphi=(L(J).*rho.*sign(cos(phi)).*sin(phi))/2./(upsi+(upsi==0)).*((((rho.*cos(phi)).^2)>=(L(J).^2/4)))+~(((rho.*cos(phi)).^2)>=(L(J).^2/4)).*rho.*sign(sin(phi)).*cos(phi);
dupsi_dL=(L(J)/2-rho.*abs(cos(phi)))./2./(upsi+(upsi==0)).*((((rho.*cos(phi)).^2)>=(L(J).^2/4))&upsi~=0);
switch p.method
    case 'MMC'
        alp=p.alp;
        epsi=p.epsi;
        bet=p.bet;
        chi0=1-(4*upsi.^2./h(J).^2).^alp;
        dchi0_dh=8*alp*upsi.^2.*(4*upsi.^2./h(J).^2).^(alp-1)./h.^3;
        dchi0_dupsi=-8*alp*upsi.*(4*upsi.^2./h(J).^2).^(alp-1)./h.^2;
        [chi,dchi]=Aggregation_Pi(chi0,p);
        dchi_dh=(dchi0_dh.*dchi);
        dchi_dupsi=(dchi0_dupsi.*dchi);
        chi(chi<=-1e6)=-1e6;
        W=(chi>epsi)+(chi<=epsi&chi>=-epsi).*(3/4*(1-bet)*(chi/epsi-chi.^3/3/epsi^3)+(1+bet)/2)+(chi<-epsi)*bet;
        dW_dchi=-3/4*(1/epsi-chi.^2/epsi^3).*(bet-1).*(abs(chi)<epsi);
        dW_dupsi=repmat(dW_dchi,size(dchi_dh,1),1).*dchi_dupsi;
        dW_dh=repmat(dW_dchi,size(dchi_dh,1),1).*dchi_dh;
        dW_dX=dW_dupsi.*(dupsi_dphi.*dphi_dX+dupsi_drho.*drho_dX);
        dW_dY=dW_dupsi.*(dupsi_dphi.*dphi_dY+dupsi_drho.*drho_dY);
        dW_dL=dW_dupsi.*dupsi_dL;
        dW_dT=dW_dupsi.*dupsi_dphi.*dphi_dT;
    case 'GP'
        deltamin=p.deltamin;
        r=p.r;
        zetavar=upsi-h(J)/2;
        dzetavar_dupsi=ones(size(upsi));
        dzetavar_dh=-0.5*ones(size(J));
        deltaiel=(1/pi/r^2*(r^2*acos(zetavar/r)-zetavar.*sqrt(r^2-zetavar.^2))).*(abs(zetavar)<=r)+((zetavar<-r));
        ddetlaiel_dzetavar=(-2*sqrt(r^2-zetavar.^2)/pi/r^2).*(abs(zetavar)<=r);
        W=deltamin+(1-deltamin)*deltaiel;
        dW_ddeltaiel=(1-deltamin);
        dW_dh=dW_ddeltaiel*ddetlaiel_dzetavar.*dzetavar_dh;
        dW_dupsi=dW_ddeltaiel*ddetlaiel_dzetavar.*dzetavar_dupsi;
        dW_dX=dW_dupsi.*(dupsi_dphi.*dphi_dX+dupsi_drho.*drho_dX);
        dW_dY=dW_dupsi.*(dupsi_dphi.*dphi_dY+dupsi_drho.*drho_dY);
        dW_dL=dW_dupsi.*dupsi_dL;
        dW_dT=dW_dupsi.*dupsi_dphi.*dphi_dT; 
    case 'MNA'
       epsi=p.sigma; 
       ds=upsi;
       d=abs(upsi);
       l=h(J)/2-epsi/2;
       u=h(J)/2+epsi/2;
       a3= -2./((l - u).*(l.^2 - 2*l.*u + u.^2));
       a2=   (3*(l + u))./((l - u).*(l.^2 - 2*l.*u + u.^2));
       a1=    -(6*l.*u)./((l - u).*(l.^2 - 2*l.*u + u.^2));
       a0=(u.*(- u.^2 + 3*l.*u))./((l - u).*(l.^2 - 2*l.*u + u.^2));
       W=1*(d<=l)+(a3.*d.^3+a2.*d.^2+a1.*d+a0).*(d<=u&d>l);
       dW_dupsi=sign(ds).*(3*a3.*d.^2+2*a2.*d+a1).*(d<=u&d>l);
       da3_du=- 2./((l - u).^2.*(l.^2 - 2*l.*u + u.^2)) - (2*(2*l - 2*u))./((l - u).*(l.^2 - 2*l.*u + u.^2).^2);
       da2_du=3./((l - u).*(l.^2 - 2*l.*u + u.^2)) + (3*(l + u))./((l - u).^2.*(l.^2 - 2*l.*u + u.^2)) + (3*(l + u).*(2*l - 2*u))./((l - u).*(l.^2 - 2*l.*u + u.^2).^2);
       da1_du=- (6*l)./((l - u).*(l.^2 - 2*l.*u + u.^2)) - (6*l.*u)./((l - u).^2.*(l.^2 - 2*l.*u + u.^2)) - (6*l.*u.*(2*l - 2*u))./((l - u).*(l.^2 - 2*l.*u + u.^2).^2);
       da0_du=(- u.^2 + 3*l.*u)./((l - u).*(l.^2 - 2*l.*u + u.^2)) + (u.*(- u.^2 + 3*l.*u))./((l - u).^2.*(l.^2 - 2*l.*u + u.^2)) + (u.*(3*l - 2*u))./((l - u).*(l.^2 - 2*l.*u + u.^2)) + (u.*(- u.^2 + 3*l.*u).*(2*l - 2*u))./((l - u).*(l.^2 - 2*l.*u + u.^2).^2);
       dWf_du=(da3_du.*d.^3+da2_du.*d.^2+da1_du.*d+da0_du).*(d<=u&d>l);
       da3_dl=2./((l - u).^2.*(l.^2 - 2*l.*u + u.^2)) + (2*(2*l - 2*u))./((l - u).*(l.^2 - 2*l.*u + u.^2).^2);
       da2_dl= 3./((l - u).*(l.^2 - 2*l.*u + u.^2)) - (3*(l + u))./((l - u).^2.*(l.^2 - 2*l.*u + u.^2)) - (3*(l + u).*(2*l - 2*u))./((l - u).*(l.^2 - 2*l.*u + u.^2).^2);
       da1_dl=    (6*l.*u)./((l - u).^2.*(l.^2 - 2*l.*u + u.^2)) - (6*u)./((l - u).*(l.^2 - 2*l.*u + u.^2)) + (6*l.*u.*(2*l - 2*u))./((l - u).*(l.^2 - 2*l.*u + u.^2).^2);
       da0_dl= (3*u.^2)./((l - u).*(l.^2 - 2*l.*u + u.^2)) - (u.*(- u.^2 + 3*l.*u))./((l - u).^2.*(l.^2 - 2*l.*u + u.^2)) - (u.*(- u.^2 + 3*l.*u).*(2*l - 2*u))./((l - u).*(l.^2 - 2*l.*u + u.^2).^2);
       dWf_dl=(da3_dl.*d.^3+da2_dl.*d.^2+da1_dl.*d+da0_dl).*(d<=u&d>l);
       dW_dh=0.5*sign(ds).*(dWf_du+dWf_dl);
        dW_dX=dW_dupsi.*(dupsi_dphi.*dphi_dX+dupsi_drho.*drho_dX);
        dW_dY=dW_dupsi.*(dupsi_dphi.*dphi_dY+dupsi_drho.*drho_dY);
        dW_dL=dW_dupsi.*dupsi_dL;
        dW_dT=dW_dupsi.*dupsi_dphi.*dphi_dT;  
end
end

function [Wa,dWa]=Aggregation_Pi(z,p)
% function that make the aggregation of the value z and also compute
% sensitivities
zm=repmat(max(z),size(z,1),1);
ka=p.ka;
switch p.aggregation
    case 'p-norm'
        zp=p.zp;
        zm=zm+zp;
        z=z+zp;
        Wa=exp(zm(1,:)).*(sum((z./exp(zm)).^ka,1)).^(1/ka)-zp;
        dWa=(z./exp(zm)).^(ka-1).*repmat((sum((z./exp(zm)).^ka,1)).^(1/ka-1),size(z,1),1);
    case 'p-mean'
        zp=p.zp;
        zm=zm+zp;
        z=z+zp;
        Wa=exp(zm(1,:)).*(mean((z./exp(zm)).^ka,1)).^(1/ka)-zp;
        dWa=1/size(z,1)^(1/ka)*(z./exp(zm)).^(ka-1).*repmat((sum((z./exp(zm)).^ka,1)).^(1/ka-1),size(z,1),1);
    case 'KS'
        Wa=zm(1,:)+1/ka*log(sum(exp(ka*(z-zm)),1));
        dWa=exp(ka*(z-zm))./repmat(sum(exp(ka*(z-zm)),1),size(z,1),1);
    case 'KSl'
        Wa=zm(1,:)+1/ka*log(mean(exp(ka*(z-zm)),1));
        dWa=exp(ka*(z-zm))./repmat(sum(exp(ka*(z-zm)),1),size(z,1),1);
    case 'IE'
        Wa=sum(z.*exp(ka*(z-zm)))./sum(exp(ka*(z-zm)),1);
        dWa=((exp(ka*(z-zm))+ka*z.*exp(ka*(z-zm))).*repmat(sum(exp(ka*(z-zm)),1),size(z,1),1)-repmat(sum(z.*exp(ka*(z-zm)),1),size(z,1),1)*ka.*exp(ka*(z-zm)))./repmat(sum(exp(ka*(z-zm)),1).^2,size(z,1),1);
end
end

function [E,dE_ddelta,dE_dm]=model_updateM(delta,p,X)
m=X(6:6:end);
nc=length(m);
m=repmat(m(:),1,size(delta,2));
switch p.method
    case 'MMC'
        %update the Young Modulus on the base of delta
        E=p.E0*delta;
        dE_ddelta=p.E0*ones(size(delta));  
        dE_ddelta=repmat(dE_ddelta,size(m,1),1);
        dE_dm=0*m;
    case 'GP'
        hatdelta=delta.*m.^p.gammac;
        [E,dE_dhatdelta]=Aggregation_Pi(hatdelta,p);
        if p.saturation
        [E,ds]=smooth_sat(E,p,nc);
        dE_dhatdelta=ds.*dE_dhatdelta;
        end
        E=E.*p.E0;
        dhatdelta_ddelta=m.^p.gammac;
        dhatdelta_dm=p.gammac*delta.*m.^(p.gammac-1);
        dE_ddelta=p.E0*dhatdelta_ddelta.*dE_dhatdelta;
        dE_dm=p.E0*dE_dhatdelta.*dhatdelta_dm;
    case 'MNA'     
        hatdelta=delta.*m.^p.gammac;
        [rho,drho_dhatdelta]=Aggregation_Pi(hatdelta,p);
        if p.saturation
        [rho,ds]=smooth_sat(rho,p,nc);
        drho_dhatdelta=ds.*drho_dhatdelta;
        end
        E=rho.^p.penalty.*(p.E0-p.Emin)+p.Emin;
        dhatdelta_ddelta=m.^p.gammac;
        dhatdelta_dm=p.gammac*delta.*m.^(p.gammac-1);
        dE_ddelta=p.penalty*(p.E0-p.Emin)*dhatdelta_ddelta.*drho_dhatdelta.*rho.^(p.penalty-1);
        dE_dm=p.penalty*(p.E0-p.Emin)*dhatdelta_dm.*drho_dhatdelta.*rho.^(p.penalty-1);

end
end

function [rho,drho_ddelta,drho_dm]=model_updateV(delta,p,X)
m=X(6:6:end);
nc=length(m);
m=repmat(m(:),1,size(delta,2));
switch p.method
    case 'MMC'
        %update the Young Modulus on the base of delta
        rho=delta;
        drho_ddelta=ones(size(delta));
        drho_ddelta=repmat(drho_ddelta,size(m,1),1);
        drho_dm=0*m;
    case 'GP'
        hatdelta=delta.*m.^p.gammav;
        [rho,drho_dhatdelta]=Aggregation_Pi(hatdelta,p);
        if p.saturation
        [rho,ds]=smooth_sat(rho,p,nc);
        drho_dhatdelta=ds.*drho_dhatdelta;
        end
        dhatdelta_ddelta=m.^p.gammav;
        dhatdelta_dm=p.gammav*delta.*m.^(p.gammav-1);
        drho_ddelta=dhatdelta_ddelta.*drho_dhatdelta;
        drho_dm=drho_dhatdelta.*dhatdelta_dm;
    case 'MNA'
        hatdelta=delta.*m.^p.gammav;
        [rho,drho_dhatdelta]=Aggregation_Pi(hatdelta,p);
        if p.saturation
        [rho,ds]=smooth_sat(rho,p,nc);
        drho_dhatdelta=ds.*drho_dhatdelta;
        end
        dhatdelta_ddelta=m.^p.gammav;
        dhatdelta_dm=p.gammav*delta.*m.^(p.gammav-1);
        drho_ddelta=dhatdelta_ddelta.*drho_dhatdelta;
        drho_dm=drho_dhatdelta.*dhatdelta_dm;
end
end

function [s,ds]=smooth_sat(y,p,nc)
switch p.aggregation
     case 'p-norm'
       xt=1;
    case 'p-mean'
       xt=(((nc-1)*p.zp^p.ka+(1+p.zp)^p.ka)/nc)^(1/p.ka)-p.zp;
    case 'KS'
         xt=1;
    case 'KSl'
         xt=1+1/p.ka*log((1+(nc-1)*exp(-p.ka))/nc);
    case 'IE'
        xt=1+1/p.ka*log((1+(nc-1)*exp(-p.ka))/nc);
end
pp=100;
s0=-log(exp(-pp)+1.0./(exp((pp.*0)./xt)+1.0))./pp;
s=@(xs,a,pa)(-log(exp(-pa)+1.0./(exp((pa.*xs)./a)+1.0))./pa-s0)/(1-s0);
ds=@(xs,a,pa)(exp((pa.*xs)./a).*1.0./(exp((pa.*xs)./a)+1.0).^2)./(a.*(exp(-pa)+1.0./(exp((pa.*xs)./a)+1.0)))/(1-s0);
% syms a xs
s=s(y,xt,pp);
ds=ds(y,xt,pp);
% s=((xt-2)/xt^3*y.^3+(3-2*xt)/xt^2*y.^2+y).*(y<=xt)+~(y<=xt);
% ds=(3*(xt-2)/xt^3*y.^2+2*(3-2*xt)/xt^2*y+ones(size(y))).*(y<=xt);
end

%-------------------------------------------------------
%    This is the file mmasub.m
%
function [xmma,ymma,zmma,lam,xsi,eta,mu,zet,s,low,upp] = ...
mmasub(m,n,iter,xval,xmin,xmax,xold1,xold2, ...
f0val,df0dx,fval,dfdx,low,upp,a0,a,c,d)
%
%    Version September 2007 (and a small change August 2008)
%
%    Krister Svanberg <krille@math.kth.se>
%    Department of Mathematics, SE-10044 Stockholm, Sweden.
%
%    This function mmasub performs one MMA-iteration, aimed at
%    solving the nonlinear programming problem:
%         
%      Minimize  f_0(x) + a_0*z + sum( c_i*y_i + 0.5*d_i*(y_i)^2 )
%    subject to  f_i(x) - a_i*z - y_i <= 0,  i = 1,...,m
%                xmin_j <= x_j <= xmax_j,    j = 1,...,n
%                z >= 0,   y_i >= 0,         i = 1,...,m
%*** INPUT:
%
%   m    = The number of general constraints.
%   n    = The number of variables x_j.
%  iter  = Current iteration number ( =1 the first time mmasub is called).
%  xval  = Column vector with the current values of the variables x_j.
%  xmin  = Column vector with the lower bounds for the variables x_j.
%  xmax  = Column vector with the upper bounds for the variables x_j.
%  xold1 = xval, one iteration ago (provided that iter>1).
%  xold2 = xval, two iterations ago (provided that iter>2).
%  f0val = The value of the objective function f_0 at xval.
%  df0dx = Column vector with the derivatives of the objective function
%          f_0 with respect to the variables x_j, calculated at xval.
%  fval  = Column vector with the values of the constraint functions f_i,
%          calculated at xval.
%  dfdx  = (m x n)-matrix with the derivatives of the constraint functions
%          f_i with respect to the variables x_j, calculated at xval.
%          dfdx(i,j) = the derivative of f_i with respect to x_j.
%  low   = Column vector with the lower asymptotes from the previous
%          iteration (provided that iter>1).
%  upp   = Column vector with the upper asymptotes from the previous
%          iteration (provided that iter>1).
%  a0    = The constants a_0 in the term a_0*z.
%  a     = Column vector with the constants a_i in the terms a_i*z.
%  c     = Column vector with the constants c_i in the terms c_i*y_i.
%  d     = Column vector with the constants d_i in the terms 0.5*d_i*(y_i)^2.
%     
%*** OUTPUT:
%
%  xmma  = Column vector with the optimal values of the variables x_j
%          in the current MMA subproblem.
%  ymma  = Column vector with the optimal values of the variables y_i
%          in the current MMA subproblem.
%  zmma  = Scalar with the optimal value of the variable z
%          in the current MMA subproblem.
%  lam   = Lagrange multipliers for the m general MMA constraints.
%  xsi   = Lagrange multipliers for the n constraints alfa_j - x_j <= 0.
%  eta   = Lagrange multipliers for the n constraints x_j - beta_j <= 0.
%   mu   = Lagrange multipliers for the m constraints -y_i <= 0.
%  zet   = Lagrange multiplier for the single constraint -z <= 0.
%   s    = Slack variables for the m general MMA constraints.
%  low   = Column vector with the lower asymptotes, calculated and used
%          in the current MMA subproblem.
%  upp   = Column vector with the upper asymptotes, calculated and used
%          in the current MMA subproblem.
%
% epsimin = sqrt(m+n)*10^(-9);
epsimin = 10^(-7);
raa0 = 0.00001;
% raa0 = 0.01;
% move = 1.0;
% albefa = 0.4;
albefa = 0.1;
% asyinit = 0.1;
asyinit = 0.01;
asyincr = 1.2;

% asyincr = 0.8;
asydecr = 0.4;
eeen = ones(n,1);
eeem = ones(m,1);
zeron = zeros(n,1);

% Calculation of the asymptotes low and upp :
if iter < 2.5
    move=1;
  low = xval - asyinit*(xmax-xmin);
  upp = xval + asyinit*(xmax-xmin);
else
  move=0.5;
  zzz = (xval-xold1).*(xold1-xold2);
  factor = eeen;
  factor(find(zzz > 0)) = asyincr;
  factor(find(zzz < 0)) = asydecr;
  low = xval - factor.*(xold1 - low);
  upp = xval + factor.*(upp - xold1);
  lowmin = xval - 0.1*(xmax-xmin);
  lowmax = xval - 0.0001*(xmax-xmin);
  uppmin = xval + 0.0001*(xmax-xmin);
  uppmax = xval + 0.1*(xmax-xmin);
  low = max(low,lowmin);
  low = min(low,lowmax);
  upp = min(upp,uppmax);
  upp = max(upp,uppmin);
end

% Calculation of the bounds alfa and beta :

zzz1 = low + albefa*(xval-low);
zzz2 = xval - move*(xmax-xmin);
zzz  = max(zzz1,zzz2);
alfa = max(zzz,xmin);
zzz1 = upp - albefa*(upp-xval);
zzz2 = xval + move*(xmax-xmin);
zzz  = min(zzz1,zzz2);
beta = min(zzz,xmax);

% Calculations of p0, q0, P, Q and b.

xmami = xmax-xmin;
xmamieps = 0.00001*eeen;
xmami = max(xmami,xmamieps);
xmamiinv = eeen./xmami;
ux1 = upp-xval;
ux2 = ux1.*ux1;
xl1 = xval-low;
xl2 = xl1.*xl1;
uxinv = eeen./ux1;
xlinv = eeen./xl1;
%
p0 = zeron;
q0 = zeron;
p0 = max(df0dx,0);
q0 = max(-df0dx,0);
%p0(find(df0dx > 0)) = df0dx(find(df0dx > 0));
%q0(find(df0dx < 0)) = -df0dx(find(df0dx < 0));
pq0 = 0.001*(p0 + q0) + raa0*xmamiinv;
p0 = p0 + pq0;
q0 = q0 + pq0;
p0 = p0.*ux2;
q0 = q0.*xl2;
%
P = sparse(m,n);
Q = sparse(m,n);
P = max(dfdx,0);
Q = max(-dfdx,0);
%P(find(dfdx > 0)) = dfdx(find(dfdx > 0));
%Q(find(dfdx < 0)) = -dfdx(find(dfdx < 0));
PQ = 0.001*(P + Q) + raa0*eeem*xmamiinv.';
P = P + PQ;
Q = Q + PQ;
P = P * spdiags(ux2,0,n,n);
Q = Q * spdiags(xl2,0,n,n);
b = P*uxinv + Q*xlinv - fval(:);
%
%%% Solving the subproblem by a primal-dual Newton method
[xmma,ymma,zmma,lam,xsi,eta,mu,zet,s] = ...
subsolv(m,n,epsimin,low,upp,alfa,beta,p0,q0,P,Q,a0,a,b,c,d,iter);
end

%-------------------------------------------------------------
%    This is the file subsolv.m
%
%    Version Dec 2006.
%    Krister Svanberg <krille@math.kth.se>
%    Department of Mathematics, KTH,
%    SE-10044 Stockholm, Sweden.
%
function [xmma,ymma,zmma,lamma,xsimma,etamma,mumma,zetmma,smma] = ...
subsolv(m,n,epsimin,low,upp,alfa,beta,p0,q0,P,Q,a0,a,b,c,d,outiter);
%
% This function subsolv solves the MMA subproblem:
%         
% minimize   SUM[ p0j/(uppj-xj) + q0j/(xj-lowj) ] + a0*z +
%          + SUM[ ci*yi + 0.5*di*(yi)^2 ],
%
% subject to SUM[ pij/(uppj-xj) + qij/(xj-lowj) ] - ai*z - yi <= bi,
%            alfaj <=  xj <=  betaj,  yi >= 0,  z >= 0.
%        
% Input:  m, n, low, upp, alfa, beta, p0, q0, P, Q, a0, a, b, c, d.
% Output: xmma,ymma,zmma, slack variables and Lagrange multiplers.
%
een = ones(n,1);
eem = ones(m,1);
epsi = 1;
epsvecn = epsi*een;
epsvecm = epsi*eem;
x = 0.5*(alfa+beta);
y = eem;
z = 1;
lam = eem;
xsi = een./(x-alfa);
xsi = max(xsi,een);
eta = een./(beta-x);
eta = max(eta,een);
mu  = max(eem,0.5*c);
zet = 1;
s = eem;
itera = 0;

iii=0;

  i1=0;  
  
  
while epsi > epsimin
    
    i1=i1+1;
    
  epsvecn = epsi*een;
  epsvecm = epsi*eem;
  ux1 = upp-x;
  xl1 = x-low;
  ux2 = ux1.*ux1;
  xl2 = xl1.*xl1;
  uxinv1 = een./ux1;
  xlinv1 = een./xl1;
  plam = p0 + P'*lam ;  
  qlam = q0 + Q'*lam ;
  gvec = P*uxinv1 + Q*xlinv1;    
  dpsidx = plam./ux2 - qlam./xl2 ;
  rex = dpsidx - xsi + eta;
  rey = c + d.*y - mu - lam;
  rez = a0 - zet - a'*lam;
  relam = gvec - a*z - y + s - b;
  rexsi = xsi.*(x-alfa) - epsvecn;
  reeta = eta.*(beta-x) - epsvecn;
  remu = mu.*y - epsvecm;
  rezet = zet*z - epsi;
  res = lam.*s - epsvecm;  
  residu1 = [rex' rey' rez]';
  residu2 = [relam' rexsi' reeta' remu' rezet res']';
  residu = [residu1' residu2']';
  residunorm = sqrt(residu'*residu);
  residumax = max(abs(residu));
  ittt = 0;    
   
  i2=0;
  
  while residumax > 0.9*epsi & ittt < 500
      
      i2=i2+1;     
      
    ittt=ittt + 1;
    itera=itera + 1;
    ux1 = upp-x;
    xl1 = x-low;
    ux2 = ux1.*ux1;
    xl2 = xl1.*xl1;
    ux3 = ux1.*ux2;
    xl3 = xl1.*xl2;
    uxinv1 = een./ux1;
    xlinv1 = een./xl1;
    uxinv2 = een./ux2;
    xlinv2 = een./xl2;
    plam = p0 + P'*lam ;
    qlam = q0 + Q'*lam ;
    gvec = P*uxinv1 + Q*xlinv1;    
    GG = P*spdiags(uxinv2,0,n,n) - Q*spdiags(xlinv2,0,n,n);
    dpsidx = plam./ux2 - qlam./xl2 ;
    delx = dpsidx - epsvecn./(x-alfa) + epsvecn./(beta-x);
    dely = c + d.*y - lam - epsvecm./y;
    delz = a0 - a'*lam - epsi/z;
    dellam = gvec - a*z - y - b + epsvecm./lam;
    diagx = plam./ux3 + qlam./xl3;
    diagx = 2*diagx + xsi./(x-alfa) + eta./(beta-x);
    diagxinv = een./diagx;
    diagy = d + mu./y;
    diagyinv = eem./diagy;
    diaglam = s./lam;
    diaglamyi = diaglam+diagyinv;    
    if m > n
      blam = dellam + dely./diagy - GG*(delx./diagx);
      bb = [blam' delz]';
      Alam = spdiags(diaglamyi,0,m,m) + GG*spdiags(diagxinv,0,n,n)*GG';
      AA = [Alam     a
            a'    -zet/z ];
      solut = AA\bb;
      dlam = solut(1:m);
      dz = solut(m+1);
      dx = -delx./diagx - (GG'*dlam)./diagx;
    else
      diaglamyiinv = eem./diaglamyi;
      dellamyi = dellam + dely./diagy;
      Axx = spdiags(diagx,0,n,n) + GG'*spdiags(diaglamyiinv,0,m,m)*GG;
      azz = zet/z + a'*(a./diaglamyi);
      axz = -GG'*(a./diaglamyi);
      bx = delx + GG'*(dellamyi./diaglamyi);
      bz  = delz - a'*(dellamyi./diaglamyi);
      AA = [Axx   axz
            axz'  azz ];
      bb = [-bx' -bz]';
      solut = AA\bb;
      dx  = solut(1:n);
      dz = solut(n+1);
      dlam = (GG*dx)./diaglamyi - dz*(a./diaglamyi) + dellamyi./diaglamyi;
    end
%
    dy = -dely./diagy + dlam./diagy;
    dxsi = -xsi + epsvecn./(x-alfa) - (xsi.*dx)./(x-alfa);
    deta = -eta + epsvecn./(beta-x) + (eta.*dx)./(beta-x);
    dmu  = -mu + epsvecm./y - (mu.*dy)./y;
    dzet = -zet + epsi/z - zet*dz/z;
    ds   = -s + epsvecm./lam - (s.*dlam)./lam;
    xx  = [ y'  z  lam'  xsi'  eta'  mu'  zet  s']';
    dxx = [dy' dz dlam' dxsi' deta' dmu' dzet ds']';
    
    stepxx = -1.01*dxx./xx;
    stmxx  = max(stepxx);
    stepalfa = -1.01*dx./(x-alfa);
    stmalfa = max(stepalfa);
    stepbeta = 1.01*dx./(beta-x);
    stmbeta = max(stepbeta);
    stmalbe  = max(stmalfa,stmbeta);
    stmalbexx = max(stmalbe,stmxx);
    stminv = max(stmalbexx,1);
    steg = 1/stminv;
%
    xold   =   x;
    yold   =   y;
    zold   =   z;
    lamold =  lam;
    xsiold =  xsi;
    etaold =  eta;
    muold  =  mu;
    zetold =  zet;
    sold   =   s;
%
    itto = 0;
    resinew = 2*residunorm;
    i3=0;
    
    while resinew > residunorm & itto < 50
        
        i3=i3+1; 
        
    itto = itto+1;
    x   =   xold + steg*dx;
    y   =   yold + steg*dy;
    z   =   zold + steg*dz;
        
    iii = iii + 1;
    
    lam = lamold + steg*dlam;
    xsi = xsiold + steg*dxsi;
    eta = etaold + steg*deta;
    mu  = muold  + steg*dmu;
    zet = zetold + steg*dzet;
    s   =   sold + steg*ds;
    ux1 = upp-x;
    xl1 = x-low;
    ux2 = ux1.*ux1;
    xl2 = xl1.*xl1;
    uxinv1 = een./ux1;
    xlinv1 = een./xl1;
    plam = p0 + P'*lam ;
    qlam = q0 + Q'*lam ;
    gvec = P*uxinv1 + Q*xlinv1;    
    dpsidx = plam./ux2 - qlam./xl2 ;
    rex = dpsidx - xsi + eta;
    rey = c + d.*y - mu - lam;
    rez = a0 - zet - a'*lam;
    relam = gvec - a*z - y + s - b;
    rexsi = xsi.*(x-alfa) - epsvecn;
    reeta = eta.*(beta-x) - epsvecn;
    remu = mu.*y - epsvecm;
    rezet = zet*z - epsi;
    res = lam.*s - epsvecm;
    residu1 = [rex' rey' rez]';
    residu2 = [relam' rexsi' reeta' remu' rezet res']';
    residu = [residu1' residu2']';
    resinew = sqrt(residu'*residu);
    steg = steg/2;  
    end
    residunorm=resinew;
  residumax = max(abs(residu));
  steg = 2*steg;
  end

  
  
  
epsi = 0.1*epsi;
end
xmma   =   x;
ymma   =   y;
zmma   =   z;
lamma =  lam;
xsimma =  xsi;
etamma =  eta;
mumma  =  mu;
zetmma =  zet;
smma   =   s;
%-------------------------------------------------------------
end

%---------------------------------------------------------------------
%  This is the file kktcheck.m
%  Version Dec 2006.
%  Krister Svanberg <krille@math.kth.se>
%
function[residu,residunorm,residumax] = ...
kktcheck(m,n,x,y,z,lam,xsi,eta,mu,zet,s, ...
         xmin,xmax,df0dx,fval,dfdx,a0,a,c,d);
%
%  The left hand sides of the KKT conditions for the following
%  nonlinear programming problem are calculated.
%         
%      Minimize  f_0(x) + a_0*z + sum( c_i*y_i + 0.5*d_i*(y_i)^2 )
%    subject to  f_i(x) - a_i*z - y_i <= 0,  i = 1,...,m
%                xmax_j <= x_j <= xmin_j,    j = 1,...,n
%                z >= 0,   y_i >= 0,         i = 1,...,m
%*** INPUT:
%
%   m    = The number of general constraints.
%   n    = The number of variables x_j.
%   x    = Current values of the n variables x_j.
%   y    = Current values of the m variables y_i.
%   z    = Current value of the single variable z.
%  lam   = Lagrange multipliers for the m general constraints.
%  xsi   = Lagrange multipliers for the n constraints xmin_j - x_j <= 0.
%  eta   = Lagrange multipliers for the n constraints x_j - xmax_j <= 0.
%   mu   = Lagrange multipliers for the m constraints -y_i <= 0.
%  zet   = Lagrange multiplier for the single constraint -z <= 0.
%   s    = Slack variables for the m general constraints.
%  xmin  = Lower bounds for the variables x_j.
%  xmax  = Upper bounds for the variables x_j.
%  df0dx = Vector with the derivatives of the objective function f_0
%          with respect to the variables x_j, calculated at x.
%  fval  = Vector with the values of the constraint functions f_i,
%          calculated at x.
%  dfdx  = (m x n)-matrix with the derivatives of the constraint functions
%          f_i with respect to the variables x_j, calculated at x.
%          dfdx(i,j) = the derivative of f_i with respect to x_j.
%   a0   = The constants a_0 in the term a_0*z.
%   a    = Vector with the constants a_i in the terms a_i*z.
%   c    = Vector with the constants c_i in the terms c_i*y_i.
%   d    = Vector with the constants d_i in the terms 0.5*d_i*(y_i)^2.
%     
%*** OUTPUT:
%
% residu     = the residual vector for the KKT conditions.
% residunorm = sqrt(residu'*residu).
% residumax  = max(abs(residu)).
%
rex   = df0dx + dfdx.'*lam - xsi + eta;
rey   = c + d.*y - mu - lam;
rez   = a0 - zet - a'*lam;
relam = fval - a*z - y + s;
rexsi = xsi.*(x-xmin);
reeta = eta.*(xmax-x);
remu  = mu.*y;
rezet = zet*z;
res   = lam.*s;
%
residu1 = [rex' rey' rez]';
residu2 = [relam' rexsi' reeta' remu' rezet res']';
residu = [residu1' residu2']';
residunorm = sqrt(residu'*residu);
residumax = max(abs(residu));
%---------------------------------------------------------------------
end

function [d]=norato_bar(xi,eta,L,h)
% to be used for plot with fill function 
d=((L/2.*sqrt(xi.^2./(xi.^2+eta.^2))+sqrt(h.^2/4-L.^2/4.*eta.^2./(xi.^2+eta.^2)))...
    .*(xi.^2./(xi.^2+eta.^2)>=(L.^2./(h.^2+L.^2)))+h./2.*sqrt(1+xi.^2./(eta.^2+(eta==0)))...
    .*(~(xi.^2./(xi.^2+eta.^2)>=(L.^2./(h.^2+L.^2)))))...
    .*(xi~=0|eta~=0)+sqrt(2)/2*h.*(xi==0&eta==0);

end

function FEM_enhanceDisplay

% Enhancing the display                    
  DisplayGrid = 1;
  Size1 = 14;
  Size2 = 16;
  
 % a = axis; axis([Fmin Fmax a(3) a(4)]);
  set(gca,'XGrid','off')
  set(gca,'YGrid','off')
  set(gca,'FontName','Arial')    
  set(gca,'LineWidth',2)    
  set(gca,'FontSize',Size1)    
  set(get(gca,'XLabel'),'fontsize',Size2,'FontWeight','bold','FontName','Arial')
  %a = get(get(gca,'XLabel'),'Position');
  %a(2) = a(2) - abs(a(2))*0.01;
  %set(get(gca,'XLabel'),'Position',a);  
  set(get(gca,'YLabel'),'fontsize',Size2,'FontWeight','bold','FontName','Arial')
  set(get(gca,'Title'),'fontsize',Size2,'FontWeight','bold','FontName','Arial')
  set(gca,'Box','On')  
  set(gcf,'WindowStyle','normal')
  set(gcf,'Position',[300 240 1000 600])
  %set(findobj(gca,'Type','line'),'LineWidth',[LineWidth])
  grid on
end


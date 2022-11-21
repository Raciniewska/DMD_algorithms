
%% Randomized LA



A       = double(im2gray(imread('Einstein.jpg')));
[m,n]   = size(A);
k0      = rank(A);
colormap bone



%% Generalized Nystrom factorization (by Nakatsukasa/Karpowicz)



% Select compression level
k       = ceil(k0*.1)
p       = 2; % oversampling factor



% Sketch matrices
D       = randn(n,k);   B  = randn(m,ceil(p*k));
%D       = randi([0,1],n,ceil(k))/n;   B  = randi([0,1],m,ceil(p*k))/m;
%for i   = 1:4
%    D   = D.*randi([0,1],n,ceil(k));
%    B   = B.*randi([0,1],m,ceil(p*k));
%end



F       = A*D;          H  = A'*B;
[Q,R]   = qr(B'*F,0);   G  = pinv(B'*F);
Ar      = (F/R)*(Q'*H');



err     = norm(Ar-A,'fro')/norm(A,'fro')



[mF,nF] = size(F);
[mG,nG] = size(G);
[mH,nH] = size(H');
Afct    = zeros(mF,nF+nG+nH+n);



OPS     = mF*mG*(nF+(mG-1)) + mG*nH*(nG+(mH-1))



%Afct(1:mF,1:n)              = rescale(Ar);
Afct(1:mF,n+1:n+nF)         = rescale(F);
Afct(1:mG,n+nF+1:n+nF+nG)   = rescale(G);
Afct(1:mH,n+nF+nG+1:n+nF+nG+nH) = rescale(H');



clf
imagesc(Afct)
axis equal off
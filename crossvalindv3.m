function INDICES = crossvalindv3(N,FOLDS)
[Y, ~] = grp2idx(N);     % at this point group is numeric only
TOTAL_DATOS = numel(Y);            %CALCULA EL TOTAL DE MUESTRAS
TOT_CLASE = accumarray(Y(:),1); %CALCULA LA FRECUENCIA POR NUMERO DE CLASES
INDICES = zeros(TOTAL_DATOS,1);           %INICIALIZAMOS EL VECTOR DE INDICES
for i = 1:numel(TOT_CLASE)
    h = find(Y==i);      %SE BUSCAN LOS ELEMENTOS QUE PERTENECEN A CADA CLASE
    % compute fold id's for every observation in the  group
    q = ceil(FOLDS*(1:TOT_CLASE(i))/TOT_CLASE(i));
    % and permute them to try to balance among all groups
    pq = randperm(FOLDS);
    % randomly assign the id's to the observations of this group
    randInd = randperm(TOT_CLASE(i));
    INDICES(h(randInd))=pq(q);
end
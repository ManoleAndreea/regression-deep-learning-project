function [matrice_W_hidden,matrice_W_output,lista_erori,lista_norme,lista_timpi]=metoda_levenberg_marquardt(matrice_A,vector_e,numar_intrari,numar_neuroni_hidden,lambda_initial,numar_maxim_iteratii,prag_oprire,parametru_a,parametru_b)
% metoda levenberg-marquardt pentru antrenarea unei retele simple
% input:
%   matrice_A = datele de intrare normalizate (cu bias)
%   vector_e = etichetele reale (valorile target)
%   numar_intrari = numar caracteristici (fara bias)
%   numar_neuroni_hidden = cati neuroni sunt pe stratul ascuns
%   lambda_initial = valoarea initiala pentru lambda (stabilitate)
%   numar_maxim_iteratii = maximul de iteratii permise
%   prag_oprire = prag pentru norma gradientului ca sa oprim
%   parametru_a, parametru_b = parametrii pentru functia activare

% initializari
numar_exemple=size(matrice_A,1); % cate exemple de antrenare avem
matrice_W_hidden=0.01*randn(numar_intrari+1,numar_neuroni_hidden); % greutati initiale hidden mici
matrice_W_output=0.01*randn(numar_neuroni_hidden,1); % greutati initiale output mici

lambda_curent=lambda_initial; % initializam lambda
iteratie=0; % contor de iteratii
norma_gradient=inf; % initial norma gradientului este infinit

% prealocare memorie pentru erori, norme si timpi
lista_erori=zeros(1,numar_maxim_iteratii);
lista_norme=zeros(1,numar_maxim_iteratii);
lista_timpi=zeros(1,numar_maxim_iteratii);

% bucla principala de antrenare
while iteratie<numar_maxim_iteratii && norma_gradient>prag_oprire
    iteratie=iteratie+1;
    t_inceput=tic; % pornim cronometru

    % forward pass: calculez iesirile in hidden si output
    matrice_Z=matrice_A*matrice_W_hidden;
    matrice_H=functie1(matrice_Z,parametru_a,parametru_b); % activare
    predictie=matrice_H*matrice_W_output; % iesirea finala

    % eroare pe toate exemplele
    vector_eroare=predictie-vector_e;

    % calcul derivata functiei de activare
    matrice_derivate=functie1_derivata(matrice_Z,parametru_a,parametru_b);

    % construirea jacobianului pentru w_hidden
    jacobian_hidden=zeros(numar_exemple,(numar_intrari+1)*numar_neuroni_hidden);
    for i=1:numar_exemple
        vector_auxiliar=(matrice_W_output'.*matrice_derivate(i,:)); % 1xnumar_neuroni_hidden
        matrice_temp=matrice_A(i,:)'*vector_auxiliar; % (numar_intrari+1) x numar_neuroni_hidden
        jacobian_hidden(i,:)=matrice_temp(:)'; % vectorizat ca linie
    end

    % nu mai trebuie reshape aici pentru ca deja e bun jacobian_hidden

    % jacobian pentru w_output
    jacobian_output=matrice_H; % este simplu doar activarea

    % concatenare jacobieni
    jacobian_total=[jacobian_hidden,jacobian_output]; % toate derivatele la un loc

    % calcul corectie parametri prin formula levenberg-marquardt
    delta_parametri=-(jacobian_total'*jacobian_total+lambda_curent*eye(size(jacobian_total,2)))\(jacobian_total'*vector_eroare);

    % actualizare greutati
    matrice_W_hidden=matrice_W_hidden+reshape(delta_parametri(1:numel(matrice_W_hidden)),size(matrice_W_hidden));
    matrice_W_output=matrice_W_output+delta_parametri(numel(matrice_W_hidden)+1:end);

    % calcul norma gradientului
    norma_gradient=norm(jacobian_total'*vector_eroare);

    % salvare progres
    lista_timpi(iteratie)=toc(t_inceput);
    lista_erori(iteratie)=sum(vector_eroare.^2)/(2*numar_exemple); % functia cost (eroarea medie patratica pe 2)
    lista_norme(iteratie)=norma_gradient;

    % afisare din 100 in 100 iteratii
    if mod(iteratie,100)==0 || iteratie==1
        fprintf('iteratia %d: (eroare=%.5f - norma gradient=%.5f)\n',iteratie,lista_erori(iteratie),norma_gradient);
    end
end

% trunchiem vectorii la numarul real de iteratii
lista_erori=lista_erori(1:iteratie);
lista_norme=lista_norme(1:iteratie);
lista_timpi=cumsum(lista_timpi(1:iteratie)); % cumulam timpul de la inceput

end

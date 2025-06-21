function [matrice_W_hidden,matrice_W_output,lista_erori,lista_norme,lista_timpi]=metoda_gradient(matrice_A,vector_e,numar_intrari,numar_neuroni_hidden,rata_invatare,numar_maxim_iteratii,prag_oprire,parametru_a,parametru_b)
% metoda gradient descent batch pentru antrenarea unei retele simple
% input:
%   matrice_A = datele de intrare normalizate (cu bias)
%   vector_e = etichetele reale (valorile target)
%   numar_intrari = numar de caracteristici (fara bias)
%   numar_neuroni_hidden = cati neuroni pe stratul ascuns
%   rata_invatare = cat de mari sunt pasii de gradient descent
%   numar_maxim_iteratii = limita maxima de iteratii
%   prag_oprire = prag pentru norma gradientului ca sa opresti
%   parametru_a, parametru_b = parametrii functiei activare

% initializari
numar_exemple=size(matrice_A,1); % cate exemple de antrenare avem
matrice_W_hidden=0.01*randn(numar_intrari+1,numar_neuroni_hidden); % greutati initiale mici pentru hidden
matrice_W_output=0.01*randn(numar_neuroni_hidden,1); % greutati initiale mici pentru output

iteratie=0; % contor iteratii
norma_gradient=inf; % norma gradient initial infinit

% prealocare memorie pentru erori, norme si timpi
lista_erori=zeros(1,numar_maxim_iteratii);
lista_norme=zeros(1,numar_maxim_iteratii);
lista_timpi=zeros(1,numar_maxim_iteratii);

% bucla principala de antrenare
while iteratie<numar_maxim_iteratii && norma_gradient>prag_oprire
    iteratie=iteratie+1;
    t_inceput=tic; % pornim cronometru

    % forward pass
    matrice_Z=matrice_A*matrice_W_hidden; % intrare strat hidden
    matrice_H=functie1(matrice_Z,parametru_a,parametru_b); % activare
    predictie=matrice_H*matrice_W_output; % iesirea finala

    % eroare
    vector_eroare=predictie-vector_e;

    % calcul gradient pentru W_output
    gradient_W_output=(1/numar_exemple)*(matrice_H')*vector_eroare;

    % calcul gradient pentru W_hidden
    matrice_derivate=functie1_derivata(matrice_Z,parametru_a,parametru_b); % derivata activarii
    eroare_propagata=vector_eroare*matrice_W_output'; % backpropagare
    gradient_W_hidden=(1/numar_exemple)*(matrice_A')*(eroare_propagata.*matrice_derivate); % calcul gradient hidden

    % calcul norma gradientului total
    norma_gradient=norm([gradient_W_hidden(:);gradient_W_output(:)]);

    % update greutati
    matrice_W_hidden=matrice_W_hidden-rata_invatare*gradient_W_hidden;
    matrice_W_output=matrice_W_output-rata_invatare*gradient_W_output;

    % salvare progres
    lista_timpi(iteratie)=toc(t_inceput);
    lista_erori(iteratie)=sum(vector_eroare.^2)/(2*numar_exemple); % eroare medie patratica / 2
    lista_norme(iteratie)=norma_gradient;

    % afisare progres la fiecare 100 iteratii
    if mod(iteratie,100)==0 || iteratie==1
        fprintf('Iteratia %d: (eroare=%.5f - norma gradient=%.5f)\n',iteratie,lista_erori(iteratie),norma_gradient);
    end
end

% trunchiere la numarul real de iteratii
lista_erori=lista_erori(1:iteratie);
lista_norme=lista_norme(1:iteratie);
lista_timpi=cumsum(lista_timpi(1:iteratie)); % cumulam timpul de la iteratii

end

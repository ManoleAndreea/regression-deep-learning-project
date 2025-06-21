function [matrice_A_train,vector_e_train,matrice_A_test,vector_e_test,medie_intrari,devstd_intrari]=proceseaza_date()
% functie care citeste, normalizeaza si imparte datele yacht_hydrodynamics
% input: niciunul
% output:
%   matrice_A_train = exemple pentru antrenare (cu bias adaugat)
%   vector_e_train = targeturi pentru antrenare
%   matrice_A_test = exemple pentru testare (cu bias adaugat)
%   vector_e_test = targeturi pentru testare
%   medie_intrari = vector cu mediile caracteristicilor initiale
%   devstd_intrari = vector cu deviatia standard a caracteristicilor initiale


% fiecare exemplu reprezinta un test de laborator asupra unui model de yacht
% sunt masurate 6 caracteristici geometrice si de viteza:
%   1. lungime relativa la deplasare
%   2. volum de deplasare
%   3. raport lungime latime
%   4. raport latime adancime
%   5. coeficient blocare carena
%   6. numarul froude (viteza yachtului)
% targetul (coloana 7) este rezistenta totala la deplasare (forta rezistenta)

% citire baza de date:
baza_date=readmatrix('yacht_hydrodynamics.csv'); % citim datele din fisierul csv

% separare intrari si iesiri:
matrice_A=baza_date(:,1:end-1); % toate coloanele mai putin ultima sunt intrari
vector_e=baza_date(:,end); % ultima coloana este iesirea (rezistenta totala)

% normalizare intrari:
% aplicam standardizare Z-score: scadem media si impartim la deviatie standard
medie_intrari=mean(matrice_A); % calculam media pe fiecare caracteristica
devstd_intrari=std(matrice_A); % calculam deviatie standard pe fiecare caracteristica
matrice_A=(matrice_A-medie_intrari)./devstd_intrari; % normalizare finala

% adaugare bias:
% adaugam o coloana de 1 la final pentru bias
matrice_A=[matrice_A,ones(size(matrice_A,1),1)];

% shuffle aleator exemple:
rng(1); % setam seed pentru random ca sa fie reproducibil
idx=randperm(size(matrice_A,1)); % permutam liniile aleator
matrice_A=matrice_A(idx,:);
vector_e=vector_e(idx,:);

% impartire in antrenare si testare:
numar_total_exemple=size(matrice_A,1); % numar total de exemple
numar_exemple_train=round(0.8*numar_total_exemple); % 80% pentru antrenare

% exemple de antrenare
matrice_A_train=matrice_A(1:numar_exemple_train,:);
vector_e_train=vector_e(1:numar_exemple_train);

% exemple de testare
matrice_A_test=matrice_A(numar_exemple_train+1:end,:);
vector_e_test=vector_e(numar_exemple_train+1:end);

end

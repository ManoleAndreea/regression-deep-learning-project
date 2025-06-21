function matrice_H=functie1(matrice_Z,parametru_a,parametru_b)
% functie de activare tip sigmoid scalata si deplasata
% input:
%   matrice_Z = intrarile catre neuronii din stratul ascuns
%   parametru_a = controleaza panta sigmoidului (cat de abrupta e curba)
%   parametru_b = controleaza deplasarea pe axa x a sigmoidului
% output:
%   matrice_H = iesirile activate pentru fiecare neuron

% aplicam formula:
% H = 1 / (1 + exp(-a*(Z-b)))
% unde:
% - a este factorul de scalare (face curba mai ingusta sau mai lata)
% - b este factorul de deplasare (muta curba la stanga sau la dreapta)

matrice_H=1./(1+exp(-parametru_a*(matrice_Z-parametru_b)));

end

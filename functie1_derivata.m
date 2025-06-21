function matrice_H_deriv=functie1_derivata(matrice_Z,parametru_a,parametru_b)
% functie care calculeaza derivata functiei de activare sigmoid scalata si deplasata
% input:
%   matrice_Z = intrarile catre neuronii din stratul ascuns
%   parametru_a = factor scalare (cat de abrupta e curba sigmoidului)
%   parametru_b = factor deplasare (muta curba la stanga sau dreapta)
% output:
%   matrice_H_deriv = derivata functiei de activare pentru fiecare neuron

% calcul activare:
% mai intai calculam activarea normala folosind functia sigmoid definita
matrice_H=functie1(matrice_Z,parametru_a,parametru_b);

% formula derivata:
% derivata sigmoidului este:
% H_deriv = a * H * (1 - H)
% unde:
% - H este iesirea activata
% - (1 - H) este complementul iesirii
% - a este factorul care scalareaza derivata

matrice_H_deriv=parametru_a.*matrice_H.*(1-matrice_H);

end

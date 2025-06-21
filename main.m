% setari initiale:
a=3; % parametrul "a" pentru functia de activare (controleaza cat de abrupta e curba)
b=0; % parametrul "b" pentru functia de activare (deplasare pe axa x)
numar_neuroni_hidden=50; % numar de neuroni pe stratul ascuns
rata_invatare=0.1; % rata cu care facem pasii de gradient descent
lambda_initial=1; % lambda initial pentru levenberg-marquardt
numar_maxim_iteratii=5000; % numarul maxim de iteratii pentru antrenare
prag_oprire=1e-6; % pragul pentru norma gradientului ca sa oprim antrenarea

% procesare date:
[A_train,e_train,A_test,e_test,medie_train,devstd_train]=proceseaza_date();
numar_intrari=size(A_train,2)-1; % numar intrari = nr coloane din A minus 1 (bias)

% metoda gradient:
fprintf(' Metoda Gradient:\n\n');
[W_hidden_gradient,W_output_gradient,lista_erori_gradient,lista_norme_gradient,lista_timpi_gradient]=metoda_gradient(A_train,e_train,numar_intrari,numar_neuroni_hidden,rata_invatare,numar_maxim_iteratii,prag_oprire,a,b);


figure('Position',[100,100,1000,700]); % fereastra mare
subplot(2,2,1);
semilogy(lista_erori_gradient,'-or','LineWidth',1,'MarkerSize',1); grid on;
title('gradient: eroare vs iteratii (log)','FontSize',10);
xlabel('iteratii','FontSize',8); ylabel('eroare','FontSize',8);

subplot(2,2,2);
semilogy(lista_timpi_gradient,lista_erori_gradient,'-or','LineWidth',1,'MarkerSize',1); grid on;
title('gradient: eroare vs timp (log)','FontSize',10);
xlabel('timp [s]','FontSize',8); ylabel('eroare','FontSize',8);

subplot(2,2,3);
semilogy(lista_norme_gradient,'-ob','LineWidth',1,'MarkerSize',1); grid on;
title('gradient: norma gradientului vs iteratii (log)','FontSize',10);
xlabel('iteratii','FontSize',8); ylabel('norma gradient','FontSize',8);

subplot(2,2,4);
semilogy(lista_timpi_gradient,lista_norme_gradient,'-ob','LineWidth',1,'MarkerSize',1); grid on;
title('gradient: norma gradientului vs timp (log)','FontSize',10);
xlabel('timp [s]','FontSize',8); ylabel('norma gradient','FontSize',8);

sgtitle('performanta metoda gradient descent','FontSize',12);

% metoda levenberg-marquardt:
fprintf(' \n Metoda Levenberg-Marquardt: \n');
[W_hidden_lm,W_output_lm,lista_erori_lm,lista_norme_lm,lista_timpi_lm]=metoda_levenberg_marquardt(A_train,e_train,numar_intrari,numar_neuroni_hidden,lambda_initial,numar_maxim_iteratii,prag_oprire,a,b);


figure('Position',[100,100,1000,700]);
subplot(2,2,1);
semilogy(lista_erori_lm,'-or','LineWidth',1,'MarkerSize',1); grid on;
title('levenberg-marquardt: eroare vs iteratii (log)','FontSize',10);
xlabel('iteratii','FontSize',8); ylabel('eroare','FontSize',8);

subplot(2,2,2);
semilogy(lista_timpi_lm,lista_erori_lm,'-or','LineWidth',1,'MarkerSize',1); grid on;
title('levenberg-marquardt: eroare vs timp (log)','FontSize',10);
xlabel('timp [s]','FontSize',8); ylabel('eroare','FontSize',8);

subplot(2,2,3);
semilogy(lista_norme_lm,'-ob','LineWidth',1,'MarkerSize',1); grid on;
title('levenberg-marquardt: norma gradientului vs iteratii (log)','FontSize',10);
xlabel('iteratii','FontSize',8); ylabel('norma gradient','FontSize',8);

subplot(2,2,4);
semilogy(lista_timpi_lm,lista_norme_lm,'-ob','LineWidth',1,'MarkerSize',1); grid on;
title('levenberg-marquardt: norma gradientului vs timp (log)','FontSize',10);
xlabel('timp [s]','FontSize',8); ylabel('norma gradient','FontSize',8);

sgtitle('performanta metoda levenberg-marquardt','FontSize',12);

% comparatie finala intre metode:
figure('Position',[100,100,800,500]);
semilogy(1:length(lista_erori_gradient),lista_erori_gradient,'-b','LineWidth',2); hold on;
semilogy(1:length(lista_erori_lm),lista_erori_lm,'-r','LineWidth',2);
grid on;
xlabel('iteratii','FontSize',12);
ylabel('eroare (scala log)','FontSize',12);
title('comparatie scadere eroare: gradient vs levenberg-marquardt','FontSize',14);
legend('gradient descent','levenberg-marquardt','Location','southwest');
set(gca,'FontSize',12);

% testare pe datele de test:
% propagare cu greutatile de la gradient
Z_test_gradient=A_test*W_hidden_gradient;
H_test_gradient=functie1(Z_test_gradient,a,b);
predictie_gradient=H_test_gradient*W_output_gradient;

% propagare cu greutatile de la levenberg-marquardt
Z_test_lm=A_test*W_hidden_lm;
H_test_lm=functie1(Z_test_lm,a,b);
predictie_lm=H_test_lm*W_output_lm;

% evaluare:
% r^2
media_test=mean(e_test);
R2_gradient=1-sum((e_test-predictie_gradient).^2)/sum((e_test-media_test).^2);
R2_lm=1-sum((e_test-predictie_lm).^2)/sum((e_test-media_test).^2);

% mse
MSE_gradient=mean((e_test-predictie_gradient).^2);
MSE_lm=mean((e_test-predictie_lm).^2);

% afisare rezultate
fprintf('\nPerformanta Gradient Descent:\nR^2 = %.4f\nMSE = %.4f\n',R2_gradient,MSE_gradient);
fprintf('\nperformanta Levenberg-Marquardt:\nR^2 = %.4f\nMSE = %.4f\n',R2_lm,MSE_lm);

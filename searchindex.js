Search.setIndex({docnames:["api/distributions/AdditiveDistribution","api/distributions/BayesRule","api/distributions/CompositeDistribution","api/distributions/Himmelblau","api/distributions/Laplace","api/distributions/LinearMatrix","api/distributions/Normal","api/distributions/SourceLocation","api/distributions/Uniform","api/distributions/_AbstractDistribution","api/distributions/index","api/index","api/massmatrices/Diagonal","api/massmatrices/LBFGS","api/massmatrices/Unit","api/massmatrices/_AbstractMassMatrix","api/massmatrices/index","api/optimizers/_AbstractOptimizer","api/optimizers/gradient_descent","api/optimizers/index","api/samplers/HMC","api/samplers/RWMH","api/samplers/_AbstractSampler","api/samplers/index","api/samples","api/visualization/index","api/visualization/marginal","api/visualization/marginal_grid","api/visualization/visualize_2_dimensions","examples","examples/0.1 - Getting started","examples/0.2 - Tuning Hamiltonian Monte Carlo","examples/1 - Gaussian inverse problems - dense forward operator","examples/2 - Gaussian inverse problems - sparse forward operator","examples/3 - Separate priors per dimension","examples/4 - Creating your own inverse problem","genindex","index","py-modindex","setup"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":3,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":2,"sphinx.domains.rst":2,"sphinx.domains.std":1,"sphinx.ext.todo":2,"sphinx.ext.viewcode":1,nbsphinx:3,sphinx:56},filenames:["api/distributions/AdditiveDistribution.rst","api/distributions/BayesRule.rst","api/distributions/CompositeDistribution.rst","api/distributions/Himmelblau.rst","api/distributions/Laplace.rst","api/distributions/LinearMatrix.rst","api/distributions/Normal.rst","api/distributions/SourceLocation.rst","api/distributions/Uniform.rst","api/distributions/_AbstractDistribution.rst","api/distributions/index.rst","api/index.rst","api/massmatrices/Diagonal.rst","api/massmatrices/LBFGS.rst","api/massmatrices/Unit.rst","api/massmatrices/_AbstractMassMatrix.rst","api/massmatrices/index.rst","api/optimizers/_AbstractOptimizer.rst","api/optimizers/gradient_descent.rst","api/optimizers/index.rst","api/samplers/HMC.rst","api/samplers/RWMH.rst","api/samplers/_AbstractSampler.rst","api/samplers/index.rst","api/samples.rst","api/visualization/index.rst","api/visualization/marginal.rst","api/visualization/marginal_grid.rst","api/visualization/visualize_2_dimensions.rst","examples.rst","examples/0.1 - Getting started.ipynb","examples/0.2 - Tuning Hamiltonian Monte Carlo.ipynb","examples/1 - Gaussian inverse problems - dense forward operator.ipynb","examples/2 - Gaussian inverse problems - sparse forward operator.ipynb","examples/3 - Separate priors per dimension.ipynb","examples/4 - Creating your own inverse problem.ipynb","genindex.rst","index.rst","py-modindex.rst","setup.rst"],objects:{"":{hmc_tomography:[11,0,0,"-"]},"hmc_tomography.Distributions":{AdditiveDistribution:[0,1,1,""],BayesRule:[1,1,1,""],CompositeDistribution:[2,1,1,""],Himmelblau:[3,1,1,""],Laplace:[4,1,1,""],LinearMatrix:[5,1,1,""],Normal:[6,1,1,""],SourceLocation:[7,1,1,""],Uniform:[8,1,1,""],_AbstractDistribution:[9,1,1,""]},"hmc_tomography.Distributions.AdditiveDistribution":{add_distribution:[0,2,1,""],collapse_bounds:[0,2,1,""],corrector:[0,2,1,""]},"hmc_tomography.Distributions.CompositeDistribution":{collapse_bounds:[2,2,1,""],corrector:[2,2,1,""],enumerated_dimensions:[2,3,1,""],enumerated_dimensions_cumulative:[2,3,1,""]},"hmc_tomography.Distributions.Himmelblau":{gradient:[3,2,1,""],misfit:[3,2,1,""],temperature:[3,3,1,""]},"hmc_tomography.Distributions.Laplace":{dispersions:[4,3,1,""],gradient:[4,2,1,""],inverse_dispersions:[4,3,1,""],means:[4,3,1,""],misfit:[4,2,1,""]},"hmc_tomography.Distributions.LinearMatrix":{gradient:[5,2,1,""],misfit:[5,2,1,""]},"hmc_tomography.Distributions.Normal":{covariance:[6,3,1,""],diagonal:[6,3,1,""],gradient:[6,2,1,""],inverse_covariance:[6,3,1,""],means:[6,3,1,""],misfit:[6,2,1,""],normalization_constant:[6,3,1,""]},"hmc_tomography.Distributions.SourceLocation":{infer_velocity:[7,3,1,""]},"hmc_tomography.Distributions.Uniform":{gradient:[8,2,1,""],misfit:[8,2,1,""]},"hmc_tomography.Distributions._AbstractDistribution":{corrector:[9,2,1,""],dimensions:[9,2,1,""],generate:[9,2,1,""],gradient:[9,2,1,""],lower_bounds:[9,3,1,""],misfit:[9,2,1,""],misfit_bounds:[9,2,1,""],name:[9,3,1,""],normalized:[9,3,1,""],update_bounds:[9,2,1,""],upper_bounds:[9,3,1,""]},"hmc_tomography.MassMatrices":{Diagonal:[12,1,1,""],LBFGS:[13,1,1,""],Unit:[14,1,1,""],_AbstractMassMatrix:[15,1,1,""]},"hmc_tomography.MassMatrices.Diagonal":{generate_momentum:[12,2,1,""],kinetic_energy:[12,2,1,""],kinetic_energy_gradient:[12,2,1,""]},"hmc_tomography.MassMatrices.Unit":{generate_momentum:[14,2,1,""],kinetic_energy:[14,2,1,""],kinetic_energy_gradient:[14,2,1,""],matrix:[14,2,1,""]},"hmc_tomography.MassMatrices._AbstractMassMatrix":{kinetic_energy:[15,2,1,""],kinetic_energy_gradient:[15,2,1,""]},"hmc_tomography.Optimizers":{_AbstractOptimizer:[17,1,1,""],gradient_descent:[18,1,1,""]},"hmc_tomography.Optimizers._AbstractOptimizer":{iterate:[17,2,1,""],iterate_once:[17,2,1,""]},"hmc_tomography.Samplers":{HMC:[20,1,1,""],RWMH:[21,1,1,""],_AbstractSampler:[22,1,1,""]},"hmc_tomography.Samplers.HMC":{acceptance_rates:[20,3,1,""],amount_of_steps:[20,3,1,""],autotuning:[20,3,1,""],current_h:[20,3,1,""],current_k:[20,3,1,""],current_momentum:[20,3,1,""],learning_rate:[20,3,1,""],mass_matrix:[20,3,1,""],minimal_stepsize:[20,3,1,""],name:[20,3,1,""],proposed_h:[20,3,1,""],proposed_k:[20,3,1,""],proposed_momentum:[20,3,1,""],sample:[20,2,1,""],stepsize:[20,3,1,""],stepsizes:[20,3,1,""],target_acceptance_rate:[20,3,1,""]},"hmc_tomography.Samplers.RWMH":{acceptance_rates:[21,3,1,""],autotuning:[21,3,1,""],learning_rate:[21,3,1,""],minimal_stepsize:[21,3,1,""],sample:[21,2,1,""],stepsize:[21,3,1,""],stepsizes:[21,3,1,""],target_acceptance_rate:[21,3,1,""]},"hmc_tomography.Samplers._AbstractSampler":{accepted_proposals:[22,3,1,""],amount_of_writes:[22,3,1,""],current_model:[22,3,1,""],current_x:[22,3,1,""],dimensions:[22,3,1,""],distribution:[22,3,1,""],max_time:[22,3,1,""],name:[22,3,1,""],progressbar_refresh_rate:[22,3,1,""],proposed_model:[22,3,1,""],proposed_x:[22,3,1,""],ram_buffer:[22,3,1,""],ram_buffer_size:[22,3,1,""],samples_hdf5_dataset:[22,3,1,""],samples_hdf5_filehandle:[22,3,1,""],samples_hdf5_filename:[22,3,1,""]},"hmc_tomography.Visualization":{marginal:[26,2,1,""],marginal_grid:[27,2,1,""],visualize_2_dimensions:[28,2,1,""]},hmc_tomography:{Distributions:[10,0,0,"-"],MassMatrices:[16,0,0,"-"],Optimizers:[19,0,0,"-"],Samplers:[23,0,0,"-"],Samples:[24,1,1,""],Visualization:[25,0,0,"-"]}},objnames:{"0":["py","module","Python module"],"1":["py","class","Python class"],"2":["py","method","Python method"],"3":["py","attribute","Python attribute"]},objtypes:{"0":"py:module","1":"py:class","2":"py:method","3":"py:attribute"},terms:{"015":32,"015652":30,"025":32,"05e":[32,33],"06e":[32,33],"075":32,"0x7f0fd55bbed0":33,"0x7f0fe0b95dd0":33,"0x7f87424ff710":32,"0x7f8742650a50":32,"100":[19,20,21,30,33],"1000":32,"10000":[30,34],"100000":30,"10311":30,"11642":30,"1282":33,"2000":32,"20000":[32,35],"2005":[32,33],"2020":30,"211":30,"212":30,"250":30,"25000":30,"3000":35,"30000":32,"30e":32,"350":30,"403":[30,32,33],"44e":33,"5000":33,"55e":32,"80000":32,"848395":30,"861192":30,"95e":33,"989091":30,"abstract":[9,15,17,22,35],"boolean":[7,9,20,21],"break":11,"case":30,"class":[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,30,32,33,35],"default":[14,20,30,33,35],"final":[30,32,33],"float":[3,4,5,6,8,9,12,13,14,15,18,20,21,22,35],"function":[0,3,9,19,20,21,30,32,33,34],"import":[16,19,23,30,31,32,33,34,35,39],"int":[2,6,13,14,20,21,22,24,26,27,28,32,33],"long":[22,39],"new":[20,21,32,33],"return":[3,9,19,30,34,35],"static":23,"throw":29,"true":[7,19,20,21,23,26,27,28,30,32,33,34,35],"try":[20,21,29,30,32,33,34,35],"while":[9,32,33],And:[32,33],But:[32,33,35],For:[20,21,30,34,35],Has:[20,21],Its:35,Not:[32,33],One:[20,30,32,33,35],That:[30,35],The:[9,10,11,12,13,14,16,19,20,21,22,23,30,32,33,34],Then:[32,33],There:[30,32,33,39],These:[11,19,30,32,33],Use:[20,21],Used:4,Useful:[20,21],Uses:[20,21],Using:21,Will:[20,21],__doc__:35,__init__:[9,35],__name__:35,_abstractdistribut:[0,2,10,18,20,21,22,30,35],_abstractmassmatrix:[16,20],_abstractoptim:19,_abstractsampl:23,_hl:30,_numpi:[2,6,9,20,21,35],_union:21,about:35,abov:9,abs:[32,33],absent:[32,33],absolut:4,acceler:4,accept:[20,21,22,30],acceptance_r:[20,21],accepted_propos:22,access:39,accord:[2,20,21],accordingli:[32,33,34],account:39,accur:[32,33],accuraci:[32,33,35],achiev:[20,21],activ:39,actual:[30,32,35],adapt:[13,19,20,21],add:[0,20,21,32,33],add_distribut:0,added:[30,32,33],adding:[32,33],addit:[0,32,33],addition:[11,30,32,33,34],adjust:[20,21],after:[0,2,9,28,30,32,33],again:[30,32,33],aggres:[20,21],aggress:[20,21],ahead:35,algorithm:[9,14,16,19,20,21,23,30,32,33,35],all:[0,2,9,10,11,12,14,15,16,19,20,21,23,29,30,32,33,34],allow:[9,20,21,22,30,32,33,35],almost:34,along:20,alpha:[32,33],alphabet:37,alreadi:32,also:[7,30,32,33,35,39],alter:[3,16],alwai:[30,32,33,35],amount:[20,21,22,30,32,33,35],amount_of_step:[20,30,32],amount_of_writ:22,amplifi:35,analys:[32,33],analyt:[32,33,35],andrea:37,ani:[14,20,21,23,30,34,35],anneal:[3,35],anoth:[2,30,32,33],anymor:[32,33],anyth:[30,32,33],anytim:34,api:[30,35,37],appear:[32,33],append:[32,33],appli:[1,30,32,33],apprais:[9,23,35],appropri:[2,32,33,39],arang:[32,33],arbitrari:[21,35],arg:[0,1,2,3,4,5,6,7,8,9],argument:[11,19,21,30,32,33],around:[32,33,34],arrai:[0,2,6,8,9,19,22,30,32,33,35],arriv:34,artifici:20,ask:[30,35],assert:35,assertionerror:[20,21],assess:[30,32,33],assign:20,associ:[9,10,16,19,20,23,30],assum:[30,32,33],attempt:9,attribut:[30,32,33,35],autocorrel:30,automat:[20,21,22,34],autotun:[20,21,33],avail:[10,16,19,23,25],avoid:30,awai:33,awar:[11,32,33],axes:[32,33],axi:[32,33],back:30,backpropag:35,balanc:[20,21],bar:22,base:[0,5,9,10,15,16,17,18,19,20,21,22,23,30,32,33,35],bash:39,bay:[0,1,10,32,33],bayesian:30,bayesrul:[30,32,33],becaus:[30,32,33,35],becom:[20,21],been:[32,33],befor:[20,21,22,30,32,33,34],begin:34,behav:[32,33],behaviour:30,below:[33,34],benefici:[9,30,35],better:[32,33],between:[20,21,22,30,32,33],bfg:19,bigger:30,bin:[26,27,28,30,32,33,35],bin_sampl:[30,32,33,34,35],bit:[32,33],black:[26,27,28],bonu:35,book:[32,33],bool:[6,7,9,20,21,26,27,28],both:[9,19,23,30,32,33],bound:[0,2,8,9,30],bracket:39,bring:[32,33],buffer:[22,30],built:[35,37],burn_in:[24,32,33],busi:30,button:30,calcul:20,call:[0,2,6,9,35],can:[2,9,10,16,19,23,29,30,32,33,34,35,39],capsiz:[32,33],captur:[32,33,34],carlo:[16,22,29,30,32,33,35],caution:[20,21],cbar:[32,33],cell:[30,32,33],center:[32,33],certainti:[32,33],chain:[13,20,21,22,30,35],chang:[11,16,30,32,33,35],characterist:0,check:[32,33,35],chi:[9,22,34,35],chi_:[9,34,35],chisquar:20,choic:39,chol:[32,33],choleski:[32,33],choos:[32,33],chosen:[20,21,30],clear:34,clone:39,close:[30,32,33],closer:[32,33],cmap:[32,33],code:[11,30,32,33,39],coincident:30,collapse_bound:[0,2],collect:[20,21],color:[26,27,28,32,33],color_1d:[27,28],colorbar:[32,33],colormap_2d:[27,28],column:[0,2,6,9,19,20,21,30,32,33,35],com:39,combin:[2,30,32,33,34],command:39,comment:[32,33],commit:37,commit_hash:37,common:30,compar:[32,33],comparison:[32,33],complaint:35,complet:[30,35,37],compon:11,composit:[0,2],compositedistribut:34,comput:[4,6,8,9,15,19,20,30,32,33,35],consequ:[32,33],constitut:[11,32,33],constrain:[32,33],construct:[19,30,32,33,34,35],constructor:[9,35],contain:[0,2,3,6,8,9,20,21,22,30,35],content:37,context:30,continu:30,control:30,conveni:[9,35],converg:[16,20,30,34],convert:19,cool:35,coordin:[0,2,3,4,5,6,8,9,35],cor:[32,33],correct:[0,2,9,37],corrector:[0,2,9],correl:[21,30,32,33,34],correspond:[2,9,32,33],cost:4,could:[30,32,33,35],counter:[32,33],coupl:5,cours:34,cov:[32,33],covari:[6,30,34],cover:30,creat:[10,19,29,30,32,33,34,39],csr:33,csr_matrix:33,ctrl:30,current:[9,20,22],current_h:20,current_k:20,current_model:22,current_momentum:20,current_x:22,d_i:[32,33],d_ob:[32,33],data:[30,32,33],data_covariance_dens:[32,33],data_covariance_diagon:[32,33],data_std:[32,33],data_std_dens:[32,33],datapoint:[32,33],dataset:30,datum:[32,33],dead:39,dear:30,deciph:35,decompos:[32,33],decomposit:[32,33],decompress:39,decoupl:[32,33],decreas:[20,21,32,33],def:[9,19,35],defin:[3,9,15,30,32,33,34,35],definit:[32,33],degre:[20,21],demonstr:35,dens:29,dense_cor:[32,33],dense_cor_prior:[32,33],dense_cov:[32,33],dense_cov_prior:[32,33],dense_d_ob:[32,33],dense_mean:[32,33],dense_mean_prior:[32,33],dense_sigma:[32,33],dense_sigma_prior:[32,33],densiti:[9,32,33,35],depend:[30,32,33],depth:34,deriv:[15,35],descent:18,describ:[2,4,9,10,11,16,19,20,21,32,33,34],descript:[11,30,32,33],design:[23,30],det:[32,33],detail:[11,20,21,30],determin:[6,7,32,33],determinist:[19,20,32,33],dev:39,deviat:[4,21,30,32,33],diag:[32,33],diagnostic_mod:[20,21],diagon:[6,30],did:[32,33],didn:35,diff:[32,33],differ:[2,23,30,32,33,34,35],digit:[32,33],dim1:28,dim2:28,dimens:[0,2,3,4,6,8,9,12,13,14,20,21,22,26,27,28,29,30,32,33,35],dimension:[3,6,22,30,34,35],dimensions_list:27,dimes:2,diminish:[20,21],direct:[20,35],directli:[9,19,23,30,32,33,34,39],directori:39,discuss:[32,33,34],disk:[22,30],disp:19,disperion:[12,32,33],dispers:[4,32,33,34],distanc:[32,33],distrbut:35,distribut:[0,1,2,3,4,5,6,7,8,9,12,14,18,19,20,21,22,23,30,32,33,34,35,39],distribution_1:30,distribution_2:30,divid:[30,35],document:[30,32,33,35,37],doe:[0,2,9,12,14,30,33,35],doesn:[30,33,35],doing:[30,34,35],don:[29,30,32,33,35],done:[30,32,33],down:30,download:39,dpoint:[32,33],draw:[9,32,33,35],drive:30,dtype:33,due:30,dure:[20,21,30],each:[2,6,8,12,20,21,30,32,33,35],earthquak:[7,34],easi:[19,30,32,33],easier:[32,33],easiest:[30,35],easili:30,effect:[32,33,35],egg:39,eigendecomposit:[32,33],eigval:[32,33],either:[6,9,30],ellipsoid:[32,33],els:[32,33,35],emper:30,encod:[32,33],encount:[32,33],end:[20,32,33,34],energi:[15,20],ensur:[9,30,32,33],entri:[20,21,32,33],enumerated_dimens:2,enumerated_dimensions_cumul:2,epsilon:18,equal:[20,21,32,33],equat:[20,32,33],error:[5,32,33,35],errorbar:[32,33],escap:39,especi:[32,33],essenti:[0,20,21,30,32,33,35],estim:[32,33],etc:[22,32,33,39],even:[32,33],everi:[0,2,9,21,30,32,33,35],everyth:[32,33],everywher:35,evid:10,evolut:30,exact:[32,33],exactli:34,exampl:[30,32,33,34,35,37],except:35,exist:[20,21,30,32,33,34,35],exit:[32,33],exp:[32,33],expect:[11,12,14,30,32,33,34],experi:[32,33],experienc:30,experiment:13,expertis:23,explan:11,expon:[32,33],express:[32,33,35],extend:[32,33],extens:30,extra:[30,39],extract:19,extrem:[32,33],eye:[32,33],factor:6,fals:[9,20,21,28,33],far:37,fashion:35,fast:35,faster:30,feel:[32,33,34],few:[30,32,33,39],fichtner:37,fig:[32,33],figsiz:[26,27,30,32,33],figur:30,file:[20,21,22,24,30,32,33,34,35,39],filenam:[20,21,22,23,24,30,34],find:[11,19,20,21,30,32,33,34],fine:[30,39],finish:[11,30],finit:35,first:[28,32,33,39],fix:[32,33],float64:33,flush:30,fmt:[32,33],folder:[29,39],follow:[3,20,30,32,33,34,35,39],format:30,formatstrformatt:[32,33],formula:[32,33],forward:[5,29],found:[10,16,29,35],frac:[3,32,33,34],framework:30,free:[7,32,33,35],from:[2,9,10,16,19,20,23,29,30,32,33,34,35,37,39],ftol:19,full:[6,11,32,33],fun:35,futur:11,gaussian:[5,6,29,34],gebraad:37,gener:[0,9,16,24,29,30,32,33,35],generate_momentum:[12,14],get:[29,34,35],get_cmap:[32,33],get_ylim:[32,33],git:39,github:[29,39],give:[21,30,32,33,35],given:[3,5,9,15,19,32,33,35],going:30,good:35,got:30,gracefulli:30,gradient:[3,4,5,6,8,9,15,18,19,33,34,35],grant:39,great:[19,30,35],group:[22,34],guarante:[13,32,33,37],gui:30,h5py:30,half:30,hamilton:20,hamiltonian:[16,29,30,32,33,35],handi:30,handl:[19,22,24],happen:[32,33,35],hard:30,has:[2,22,30,32,33,34,35],have:[0,2,10,12,14,19,30,32,33,34,35,39],haven:34,hdf5:[20,21,22,30,32,33],hdf:30,hdfcompass:30,help:[30,32,33],here:[11,30,32,33,35],high:30,highli:[32,33],his:[23,30],hist:[32,33],histogram:[28,35],hmc:[0,2,9,14,16,23,30,32,33,34,35,39],hmc_instanc:[23,30],hmc_tomographi:[0,1,2,3,4,5,6,7,8,9,11,12,13,14,15,17,18,19,20,21,22,23,24,28,30,31,32,33,34,35,39],hmctomo:34,hold:30,home:[30,32,33],hope:[32,33],hour:30,how:[2,11,20,21,22,30,32,33,35],howev:[11,23,30,35],hypocent:34,ident:[32,33],identifi:30,illustr:[32,33,34,35],immedi:28,impact:[16,32,33],implement:[9,10,19,21,30,35,37],implic:10,improp:35,imshow:[32,33],includ:[11,30],increas:[32,33],indent:[32,33],independ:[12,35],index:[2,30,32,33,37],indic:[6,20,21,22,30,35],inf:9,infer:[30,32,33,34],infer_veloc:7,infinit:[32,33],influenc:[20,21,30,32,33,35],inform:[30,32,33],inherit:[10,16,19,23,35],initi:[9,20,30,33],initial_model:[20,21,32,33],initialis:23,inject:23,inplement:35,input:[19,35],inspect:[30,35],instal:37,instanc:[20,21,23,30,35],instanti:35,instead:39,intact:[30,34],integ:[20,21,22,32,33],integr:[0,2,9,20,30],interest:[30,32,33,35],interfac:[30,34],intermitt:30,interpret:[32,33],interv:[20,21],introduc:[32,33,34],intuit:[32,33],inv:[32,33],inv_data_cov:[32,33],inv_prior_cov:[32,33],invalid:[20,21,32,33],invers:[4,6,10,19,29,30],inverse_covari:6,inverse_dispers:4,invert:[32,33],investig:[28,35],invoc:9,invok:2,issu:29,issubclass:30,iter:17,iterate_onc:17,iteration_numb:[20,21],its:[9,20,30,32,33,34,35],itself:[0,2,9,32,33,35],jac:19,jointli:28,jupyt:[29,30],just:30,keep:[32,33,34],kei:39,kept:[20,21],kernel:30,keyword:21,kind:30,kinet:[15,20],kinetic_energi:[12,14,15],kinetic_energy_gradi:[12,14,15],know:[30,35,37],knowledg:[32,33,34],known:[32,33],kwarg:[0,1,2,3,4,5,6,7,8,9],label:[32,33],laplac:[32,33,34],lar:37,larg:30,larger:[20,21],larsgeb:39,larsgebraad:[30,32,33],lasso:4,last:30,leapfrog:[20,30],learning_r:[20,21],least:[4,30,34],leav:[30,32,33,34,35],left:[22,32,33],legend:[32,33],length:[21,32,33,34],less:[30,32,33],let:[30,32,33,35,37],level:[0,2],lie:[32,33],lies:22,life:[32,33],like:[32,33,35],likelihood:[5,6,30,32,33,35],limit:[6,8,30],linalg:[32,33],line:[32,33],linear:[5,32,33],linearmatrix:[32,33],linearsegmentedcolormap:[27,28],list:[2,27,30],list_of_distribut:2,littl:[30,32,33],load:30,local:[35,39],locat:7,log10:[32,33],log:[9,22,35],logarithm:[32,33,34],logic:[32,33],longer:30,look:[30,32,33,35],lot:[11,30],lower:[6,8,9,20,21,32,33],lower_bound:[6,8,9,30],luckili:30,m_1:34,m_2:34,m_true:[32,33],machin:34,made:[32,33],magnitud:33,mai:[32,33],main:[30,32,33],make:[20,21,30,32,33,35],manag:30,mani:[2,9,20,21,30,32,33,35],manual:[32,33],margin:[27,30,32,33,34],marginal_grid:[30,32,33,34,35],marker:[32,33],markov:[13,20,21,22,30,35],mass:[12,13,14,15,16,20,30,34,35],mass_matrix:[20,30,32,33,34],massmatric:[12,13,14,15,20,32,33,34],master:39,math:35,mathbf:[0,2,5,9,32,33,34,35],mathcal:[32,33],mathemat:[32,33,34],matplotlib:[27,28,30,32,33],matric:[15,32,33],matrix:[6,12,13,14,16,20,30,34,35],max:[32,33],max_determinant_chang:13,max_tim:[20,21,22,30],maxcor:19,maximum:[20,21,22],maxit:19,maxnloc:[32,33],mayb:[30,32,33],mcmc:[22,30,32,33],mean:[4,6,30,32,33,34],measur:[32,33],medium:[20,32],memori:4,messag:29,method:[0,2,4,6,8,9,10,11,15,16,19,23,25,26,27,28,30,32,33,35],metric:[12,14,16],might:[11,30,32,33,34],mind:[32,33,34],minim:[19,20,21,23,30,33],minima:[19,35],minimal_steps:[20,21],minimum:33,misfit:[3,4,5,6,8,9,19,22,34,35],misfit_bound:9,miss:[11,35,37],mixtur:9,mock:35,model:[2,4,5,6,7,8,9,20,21,22,30,32,33],modifi:[32,33],modul:[10,11,16,19,23,30,37],moment:39,momenta:[0,2,9],momentum:[0,2,9,12,14,15,20],mont:[16,22,29,30,32,33,35],more:[1,2,20,21,30,32,33,35],most:[10,30,32,33],mostli:[9,35],motion:[32,33],move:20,much:[32,33],multipl:[2,27,30],multivari:[4,6,21,30,32,33],mutlivari:30,mvn:21,my_inverse_problem_class:35,nabla_:[9,35],name:[9,11,20,22],nan:20,ndarrai:[0,2,3,4,5,6,8,9,12,13,14,15,17,20,21,22,30,35],necessarili:30,need:[9,19,20,21,30,32,33,34,35,39],neg:[20,21,22,34],never:6,next:[30,32,33],nice:30,nmean:[32,33],nois:[32,33],non:10,none:[8,9,13,19,20,21,22,32,33,35],normal:[2,9,10,21,30,32,33,34],normal_2d:30,normaliz:[32,33],normalization_const:6,note:[9,30,32,33,34,35],notebook:[29,30,32,33,34,35],noth:35,notic:30,notimplementederror:[9,35],now:[19,30,32,33,35,39],number:[9,30],number_of_vector:13,numer:[19,20],numpi:[0,2,3,4,5,6,8,9,12,13,14,15,17,20,21,22,30,31,32,33,34,35],object:[0,2,9,19,20,22,27,28,30,32,33,34,35],obs:[32,33],observ:[5,32,33,34],obtain:[32,33],occur:34,oct:30,off:[12,14,30,32,33],often:30,omit:35,onc:[30,39],one:[2,9,30,32,33,34],ones:[30,32,33,34],onli:[0,2,9,12,23,28,30,32,33,34,35],onlin:[20,21,30],online_thin:[17,20,21,30,32,33],oop:35,open:[32,33,35,39],oper:[0,2,9,29,30,35],opposit:[32,33],optim:[9,12,14,17,18,20,21,35],option:[9,19,30],order:[2,30,32,33,34,35],origin:[30,34],other:[0,9,30,32,33,35],otherwis:[32,33],our:[30,32,33,34],ourselv:35,out:[20,21,30,32,33,35],outlin:[10,16,19,23],output:[10,16,19,23,28,32,33,35],over:[30,35],overdetermin:32,overhead:[20,21],overrid:[0,2],overwrit:[20,21,30,32,33,34,35],overwrite_existing_fil:[20,21,30,32,33,34,35],own:[10,29,30],packag:[11,30,34,35,37],page:[32,33],parallel:35,paramet:[0,2,6,7,8,9,12,14,15,16,17,20,21,23,28,30,32,33,34,35],part:[9,20,21,35],partial:34,particl:[0,2,9],particular:29,pass:[2,9,20,30,32,33,35],past:[20,21],path:[20,21,22],peek:30,per:[29,30],percentag:30,perfect:[30,34],perfectli:34,perform:[12,14,16,35],permiss:30,piec:[29,35],pip:39,plai:[32,33,34],pleas:37,plot:[28,30,32,33],plt:[30,32,33],pmatrix:34,point:[32,33,35],posit:[4,20,22,32,33],possibl:[32,33],post:30,posterior:[19,28,30,32,33,34],posterior_laplac:[32,33],potenti:11,power:23,precis:[32,33,34],prefer:[32,33,34],premultipl:33,press:30,previou:[30,32,33],primarili:[32,33,34],print:[30,32,33,35],print_detail:30,prior:[29,30,35],prior_1:34,prior_2:34,prior_laplac:[32,33],prior_mean:[32,33],prior_vari:[32,33],privat:39,probabilist:[32,33],probabl:[9,22,32,33,34,35],problem:[10,29,30],process:[30,34,35],produc:13,product:[32,33],program:30,progress:[22,30],progressbar_refresh_r:22,project:29,promot:[32,33],prompt:30,proper:[32,33,39],properli:30,properti:[14,30,35],proport:[32,33],propos:[17,20,21,22,30,32,33,34,35],proposed_h:20,proposed_k:20,proposed_model:22,proposed_momentum:20,proposed_x:22,propto:[32,33],provid:[23,30,32,33,35,37],pseudo:[32,33],pull:[32,33],push:[32,33],pyplot:[28,30,32,33],python:[30,35,39],quantifi:[10,32,33],quantiti:[32,33],quickli:30,quit:[32,33],rais:[2,9,20,21,32,33,35],ram:[20,21,22,30],ram_buff:22,ram_buffer_s:[20,21,22],rand:32,randn:[32,33],random:[32,33,35],randomli:[32,33],rang:[32,33],rank:[32,33],rate:[20,21,22,30],rather:35,raw_samples_hdf:30,readi:35,readm:39,realis:[32,33],realiz:[30,32,33],reason:30,recommend:39,record:30,recycl:[32,33],reduc:[32,33,35],refer:[0,2,9,30,35,37],region:33,regular:[32,33],reject:30,relat:[9,19,29,32,33,35],relev:[32,33,34],rememb:[32,33],render:28,repo:[29,39],repres:[0,2,3,6,9,20,21,22,34,35],request:30,requir:[2,9,10,15,16,19,20,21,23,30,32,33,35,39],resampl:20,resolv:[32,33],resp:[32,33],respect:[20,35],restructur:[0,2],result:[19,32,33,34,35,39],right:[22,32,33,35],rng:[32,33],round:30,routin:[17,18],rule:[0,1,10,32,33],run:[19,23,30,39],same:[14,30,32,33,35],sampl:[9,20,21,22,23,28,29,32,33,34,35],sample_ram_buffer_s:17,sampler:[20,21,22,30,32,33,34,35],samples_0:30,samples_filenam:17,samples_hdf5_dataset:22,samples_hdf5_filehandl:22,samples_hdf5_filenam:[20,21,22],samples_posterior:[32,33],samples_posterior_gaussian:[32,33],samples_posterior_laplac:[32,33],sampling_cor:[32,33],sampling_cov:[32,33],sampling_dense_cor:[32,33],sampling_dense_cor_prior:[32,33],sampling_dense_cor_prior_laplac:[32,33],sampling_dense_cov:[32,33],sampling_dense_cov_prior:[32,33],sampling_dense_cov_prior_laplac:[32,33],sampling_dense_mean:[32,33],sampling_dense_mean_prior:[32,33],sampling_dense_mean_prior_laplac:[32,33],sampling_dense_sigma:[32,33],sampling_dense_sigma_prior:[32,33],sampling_dense_sigma_prior_laplac:[32,33],sampling_mean:[32,33],sampling_sigma:[32,33],satisfi:[20,21],save:30,scalar:[32,33],scale:[12,14,32,33,35],scatter:[32,33],scheme:[20,21],scipi:33,second:[20,21,22,28,30],section:30,see:[30,32,33,35,39],seed:[32,33],seem:[30,32,33],seismic:[32,33],self:[9,35],semidefinit:[32,33],sens:[32,33],separ:[2,6,12,29,32,33,35],separate_distribut:2,set:[2,14,20,21,30,32,33,34,35,39],set_major_formatt:[32,33],set_major_loc:[32,33],set_printopt:[32,33],set_titl:[32,33],set_xlabel:[32,33],set_xlim:[32,33],set_ylabel:[32,33],set_ylim:[32,33],setup:39,shape:[0,2,3,4,6,8,9,16,19,20,21,30,32,33,35],sharpen:35,shell:39,shorter:35,should:[2,9,19,20,21,30,32,33,35],shouldn:[32,33],show:[23,26,27,28,30,32,33,34,35],shown:[28,30],sigma:[32,33],signatur:[10,11,16,19,23],silent:[20,21,30,32,33,34,35],sim:[32,33],similar:[32,33,35],similarli:[19,35],simpl:[30,32,33,34,35,39],simpli:[9,30,35,39],simul:35,simultan:[27,32,33],singl:[6,7,32,33],size:[20,21,22,30,32,33],slice:30,slow:[20,21,30],smaller:[20,21,33],sneakili:30,snippet:30,solut:[32,33],solv:[20,32,33],solvabl:[32,33],some:[9,11,30,35],someth:[30,37],sometim:[30,34],sourc:[0,1,2,3,4,5,6,7,8,9,12,13,14,15,17,18,20,21,22,24],space:[2,4,6,8,9,30,32,33,35],spars:[29,32],sparsiti:[32,33],special:30,specif:[20,21,30,32,33],specifi:[30,35],split:2,sqrt:[32,33],squar:[32,33],ssh:39,stage:20,standard:[21,30,32,33],standardnormal1d:30,start:[20,21,29,32,33,35],starting_gradi:13,starting_model:19,starting_posit:13,state:[20,22,30],statist:[10,22,30],std:30,step:[0,2,9,20,21,30,32,33,34,35],stepsiz:[20,21,30,32,33,34,35],still:[11,12,30,32,33,35],stop:[30,35],storag:[20,21],store:[20,21,22],str:[9,20,21,22],string:[20,21,22],strong:[20,21,32,33],strongli:[32,33],structur:[32,33],studi:32,styblinskitang:35,styblisnki:35,sub:[0,2],subclass:[20,21],submodul:30,subplot:[30,32,33],subroutin:[20,21],subsequ:[34,35],subsurfac:7,subtyp:20,succe:39,suffic:[9,35],suffici:34,sum:[0,2,32,33,35],supplement:[32,33],suppli:[19,30,32,33,35],support:[32,33],suppress:[32,33],surpress:35,surpris:[32,33],symplect:20,sys:[30,32,33,34,35],system:39,take:[20,21,22,30],tarantola:[32,33],target:[12,14,18,19,20,21,23],target_acceptance_r:[20,21],technic:35,techniqu:35,temper:35,temperatur:[3,35],templat:35,term:[6,10,32,33],termin:[20,21,22,30,39],test:[32,33],text:[9,32,33,34,35],than:[20,21,30,32,33,35],thei:[30,32,33,39],them:[10,23,30],theoret:[32,33],theori:[29,32,33],therebi:16,therefor:[9,35],thi:[0,2,6,9,10,11,12,13,14,16,19,20,21,23,30,32,33,34,35,37,39],thin:[20,21,30],thing:[30,32,33,35],think:[30,35],those:30,though:[32,33],three:[32,33],through:[19,23,32,33,35],throughout:[19,32,33,34],ticker:[32,33],tight_layout:[32,33],time:[0,2,9,20,21,22,32,33,34,35],timestep:21,titl:32,tocsr:33,todens:33,todo:31,togeth:[11,29,32,33],tomographi:[30,32,33,39],too:[32,33],top:[0,2,30,37],topic:30,total:[20,30,32,33,34],toward:[20,21,32,33],trace:30,trade:[12,14,30,32,33],trajectori:[16,20],trial:35,truncat:[9,30],tryset:33,tune:[16,20,21,23,29,32,33,34],turn:[32,33,35],tutori:[10,16,37],tutorial_01_mvn:30,tutorial_01_posterior:30,tutorial_01_standard_norm:30,tutorial_1_dense_covari:32,tutorial_1_dense_covariance_with_gaussian_prior:32,tutorial_1_dense_covariance_with_laplace_prior:32,tutorial_1_diagonal_covari:32,tutorial_2_dense_covari:33,tutorial_2_dense_covariance_with_gaussian_prior:33,tutorial_2_dense_covariance_with_laplace_prior:33,tutorial_2_diagonal_covari:33,tutorial_3_composit:34,tutorial_4_empti:35,tutorial_4_styblinskitang_20:35,tutorial_4_styblinskitang_5:35,two:[0,1,2,30,32,33,34],type:[9,30,33,34],typeerror:[20,21],typic:[20,30,32,33,35],unboud:[8,35],uncertainti:[32,33],uncomput:6,uncondit:2,uncorrel:[4,6,30,32,33],understand:[29,35],uniform:[2,30,35],union:21,uniqu:21,unit:[20,30,32,33,34],unlimit:[20,21,22],unnorm:[0,1],unnormaliz:35,unscal:18,unspecifi:[20,21],unzip:39,updat:[9,11,20,21,22,32,33,34],update_bound:9,update_interv:13,upon:[0,2,9,30],upper:[6,8,9,32,33],upper_bound:[6,8,9,30],usag:4,use:[9,19,30,32,33,34,35],used:[2,9,16,19,20,21,22,28,30,32,33,35],useful:30,user:[23,30],userwarn:33,using:[7,20,21,23,28,30,32,33,34,35,39],valid:[13,32,33],valu:[3,20,21,22,30,32,33],valueerror:[2,20,21,32,33],vanilla:30,vari:[12,30],variabl:[32,33],varianc:[6,12,14,30,32,33],variat:[32,33],variou:[19,32,33,34],vector:[0,2,6,9,19,20,21,30,32,33,35],veloc:7,verif:[32,33],verifi:[30,32,33],version:[35,39],via:30,virtual:39,visual:[26,27,28,30,32,33,34,35],visualize_2_dimens:[30,34],vital:16,vmax:[32,33],vmin:[32,33],volum:[32,33],wai:[3,19,23,30,34,35],want:[30,32,33,35,39],warn:[30,32,33,34,35],wast:19,well:[30,32,33,35],went:33,were:30,what:[30,32,33,34,35],when:[2,12,14,23,30,32,33,34,35],where:[30,32,33,34],whether:[6,7,20,21,28],which:[0,2,9,20,21,22,30,32,33,34,35,39],why:35,wish:[30,35],within:[9,10,16,19,23,39],without:[32,33],won:39,work:[11,22,23,30,32,33,35,39],worri:35,wors:[32,33],would:[19,30,34,35],wouldn:35,wrap:30,write:[20,21,23,29,30],written:[20,21,22],xaxi:[32,33],xlabel:30,yaxi:[32,33],yerr:[32,33],yet:[21,22,30,32,33,35],ylabel:30,ylim:[32,33],ymax:[32,33],you:[11,19,23,29,30,32,33,34,35,37,39],your:[9,10,12,14,23,29,32,33,39],yourself:35,zero:[9,20,21,30,32,33,34,35],zip:39,zsh:39,zunino:37},titles:["AdditiveDistribution","BayesRule","CompositeDistribution","Himmelblau","Laplace","LinearMatrix","Normal","SourceLocation","Uniform","_AbstractDistribution","Distributions","API reference","Diagonal","LBFGS","Unit","_AbstractMassMatrix","MassMatrices","_AbstractOptimizer","gradient_descent","Optimizers","HMC - vanilla Hamiltonian Monte Carlo","RWMH - Random Walk Metropolis Hastings","_AbstractSampler","Samplers","Samples class","Visualization","marginal","marginal_grid","visualize_2_dimensions","Examples and tutorials","Tutorial 0.1 - Getting started","Tutorial 0.2 - Tuning Hamiltonian Monte Carlo","Tutorial 1 - Gaussian inverse problems - dense forward operator","Tutorial 2 - Gaussian inverse problems - sparse forward operator","Tutorial 3 - Separate priors per dimension","Tutorial 4 - Creating your own inverse problem","Alphabetic index","HMC Tomography","Module index","Installation"],titleterms:{"case":[32,33],"class":24,"function":35,"new":39,The:35,Using:[19,30],_abstractdistribut:9,_abstractmassmatrix:15,_abstractoptim:17,_abstractsampl:22,access:30,additivedistribut:0,all:35,alphabet:36,api:11,bay:30,bayesrul:1,big:30,brownian:35,carlo:[20,31],compositedistribut:2,conda:39,covari:[32,33],creat:35,deal:30,dens:[32,33],depend:39,develop:39,diagon:[12,32,33],dimens:34,distribut:10,empti:35,environ:39,exampl:29,first:30,forward:[32,33],gather:35,gaussian:[32,33],get:30,gradient_desc:18,hamiltonian:[20,31],hast:21,himmelblau:3,hmc:[20,37],index:[36,38],ingredi:35,instal:39,interpret:30,interrupt:30,invers:[32,33,35],investig:30,laplac:4,lbfg:13,linearmatrix:5,margin:26,marginal_grid:27,massmatric:16,matrix:[32,33],metropoli:21,misfit:30,model:35,modul:38,mont:[20,31],motion:35,multidimension:30,non:[32,33],normal:6,one:39,oper:[32,33],optim:19,option:39,own:35,packag:39,per:34,prior:[32,33,34],problem:[32,33,35],random:21,raw:30,refer:11,result:30,routin:19,rule:30,rwmh:21,sampl:[24,30],sampler:23,scipi:19,separ:34,sourceloc:7,spars:33,start:30,styblinski:35,tang:35,target:35,three:39,time:30,tomographi:37,tune:31,tutori:[29,30,31,32,33,34,35],two:39,uniform:8,unit:14,vanilla:20,visual:25,visualize_2_dimens:28,walk:21,your:[30,35]}})
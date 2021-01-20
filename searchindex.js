Search.setIndex({docnames:["api/constants","api/fitting_pipeline","api/models","api/plasticc","api/plotting","api/reference_stars","api/sn_magnitudes","api/utils/caching","api/utils/filters","api/utils/time_series","api/utils/utils","index","notebooks/lsst_filters","notebooks/notebook_summaries","notebooks/pwv_eff_on_black_body","notebooks/pwv_modeling","notebooks/simulating_lc_for_cadence","notebooks/sne_delta_mag","overview/command_line","overview/data_provenance","overview/install","overview/integrations","overview/plasticc_model"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":3,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":2,"sphinx.domains.rst":2,"sphinx.domains.std":1,"sphinx.ext.viewcode":1,nbsphinx:3,sphinx:56},filenames:["api/constants.rst","api/fitting_pipeline.rst","api/models.rst","api/plasticc.rst","api/plotting.rst","api/reference_stars.rst","api/sn_magnitudes.rst","api/utils/caching.rst","api/utils/filters.rst","api/utils/time_series.rst","api/utils/utils.rst","index.rst","notebooks/lsst_filters.nblink","notebooks/notebook_summaries.rst","notebooks/pwv_eff_on_black_body.nblink","notebooks/pwv_modeling.nblink","notebooks/simulating_lc_for_cadence.nblink","notebooks/sne_delta_mag.nblink","overview/command_line.rst","overview/data_provenance.rst","overview/install.rst","overview/integrations.rst","overview/plasticc_model.rst"],objects:{"snat_sim.models":{AbstractVariablePWVEffect:[2,1,1,""],FixedResTransmission:[2,1,1,""],ObservedCadence:[2,1,1,""],PWVModel:[2,1,1,""],SNModel:[2,1,1,""],SeasonalPWVTrans:[2,1,1,""],StaticPWVTrans:[2,1,1,""],VariablePWVTrans:[2,1,1,""],VariablePropagationEffect:[2,1,1,""]},"snat_sim.models.AbstractVariablePWVEffect":{__init__:[2,2,1,""],assumed_pwv:[2,2,1,""],propagate:[2,2,1,""]},"snat_sim.models.FixedResTransmission":{__init__:[2,2,1,""],calc_transmission:[2,2,1,""]},"snat_sim.models.ObservedCadence":{__init__:[2,2,1,""],from_plasticc:[2,2,1,""],to_sncosmo:[2,2,1,""]},"snat_sim.models.PWVModel":{__init__:[2,2,1,""],calc_airmass:[2,2,1,""],from_suominet_receiver:[2,2,1,""],pwv_los:[2,2,1,""],pwv_zenith:[2,2,1,""],seasonal_averages:[2,2,1,""]},"snat_sim.models.SNModel":{simulate_lc:[2,2,1,""],simulate_lc_fixed_snr:[2,2,1,""]},"snat_sim.models.SeasonalPWVTrans":{__init__:[2,2,1,""],assumed_pwv:[2,2,1,""],from_pwv_model:[2,2,1,""]},"snat_sim.models.StaticPWVTrans":{__init__:[2,2,1,""],propagate:[2,2,1,""],transmission_res:[2,2,1,""]},"snat_sim.models.VariablePWVTrans":{__init__:[2,2,1,""],assumed_pwv:[2,2,1,""]},"snat_sim.models.VariablePropagationEffect":{propagate:[2,2,1,""]},"snat_sim.pipeline":{FitLightCurves:[1,1,1,""],FitResultsToDisk:[1,1,1,""],FittingPipeline:[1,1,1,""],LoadPlasticcSims:[1,1,1,""],PipelineResult:[1,1,1,""],SimulateLightCurves:[1,1,1,""],SimulationToDisk:[1,1,1,""]},"snat_sim.pipeline.FitLightCurves":{__init__:[1,2,1,""],action:[1,2,1,""],fit_lc:[1,2,1,""]},"snat_sim.pipeline.FitResultsToDisk":{__init__:[1,2,1,""],action:[1,2,1,""],setup:[1,2,1,""]},"snat_sim.pipeline.FittingPipeline":{__init__:[1,2,1,""]},"snat_sim.pipeline.LoadPlasticcSims":{__init__:[1,2,1,""],action:[1,2,1,""]},"snat_sim.pipeline.PipelineResult":{__init__:[1,2,1,""],column_names:[1,2,1,""],to_csv:[1,2,1,""],to_list:[1,2,1,""]},"snat_sim.pipeline.SimulateLightCurves":{__init__:[1,2,1,""],action:[1,2,1,""],add_pwv_columns_to_table:[1,2,1,""],duplicate_plasticc_lc:[1,2,1,""]},"snat_sim.pipeline.SimulationToDisk":{__init__:[1,2,1,""],action:[1,2,1,""],setup:[1,2,1,""]},"snat_sim.plasticc":{PLaSTICC:[3,1,1,""]},"snat_sim.plasticc.PLaSTICC":{__init__:[3,2,1,""],count_light_curves:[3,2,1,""],format_data_to_sncosmo:[3,2,1,""],get_available_cadences:[3,2,1,""],get_model_headers:[3,2,1,""],iter_lc:[3,2,1,""]},"snat_sim.plotting":{plot_cosmology_fit:[4,3,1,""],plot_delta_colors:[4,3,1,""],plot_delta_mag_vs_pwv:[4,3,1,""],plot_delta_mag_vs_z:[4,3,1,""],plot_delta_mu:[4,3,1,""],plot_derivative_mag_vs_z:[4,3,1,""],plot_fitted_params:[4,3,1,""],plot_magnitude:[4,3,1,""],plot_pwv_mag_effects:[4,3,1,""],plot_residuals_on_sky:[4,3,1,""],plot_spectral_template:[4,3,1,""],plot_year_pwv_vs_time:[4,3,1,""],sci_notation:[4,3,1,""]},"snat_sim.reference_stars":{ReferenceCatalog:[5,1,1,""],ReferenceStar:[5,1,1,""],VariableCatalog:[5,1,1,""]},"snat_sim.reference_stars.ReferenceCatalog":{__init__:[5,2,1,""],average_norm_flux:[5,2,1,""],calibrate_lc:[5,2,1,""]},"snat_sim.reference_stars.ReferenceStar":{__init__:[5,2,1,""],flux:[5,2,1,""],get_available_types:[5,2,1,""],get_dataframe:[5,2,1,""],norm_flux:[5,2,1,""],to_pandas:[5,2,1,""]},"snat_sim.reference_stars.VariableCatalog":{__init__:[5,2,1,""],average_norm_flux:[5,2,1,""],calibrate_lc:[5,2,1,""]},"snat_sim.sn_magnitudes":{calc_calibration_factor_for_params:[6,3,1,""],calc_delta_mag:[6,3,1,""],calc_mu_for_model:[6,3,1,""],calc_mu_for_params:[6,3,1,""],correct_mag:[6,3,1,""],fit_mag:[6,3,1,""],get_config_pwv_vals:[6,3,1,""],tabulate_fiducial_mag:[6,3,1,""],tabulate_mag:[6,3,1,""]},"snat_sim.utils":{caching:[7,0,0,"-"],filters:[8,0,0,"-"],setup_environment:[10,3,1,""],time_series:[9,0,0,"-"]},"snat_sim.utils.caching":{Cache:[7,1,1,""],MemoryCache:[7,1,1,""]},"snat_sim.utils.caching.Cache":{__init__:[7,2,1,""]},"snat_sim.utils.caching.MemoryCache":{__init__:[7,2,1,""]},"snat_sim.utils.filters":{register_decam_filters:[8,3,1,""],register_lsst_filters:[8,3,1,""],register_sncosmo_filter:[8,3,1,""]},"snat_sim.utils.time_series":{TSUAccessor:[9,1,1,""],datetime_to_sec_in_year:[9,3,1,""]},"snat_sim.utils.time_series.TSUAccessor":{__init__:[9,2,1,""],periodic_interpolation:[9,2,1,""],resample_data_across_year:[9,2,1,""],supplemented_data:[9,2,1,""]},snat_sim:{constants:[0,0,0,"-"],models:[2,0,0,"-"],pipeline:[1,0,0,"-"],plasticc:[3,0,0,"-"],plotting:[4,0,0,"-"],reference_stars:[5,0,0,"-"],sn_magnitudes:[6,0,0,"-"],utils:[10,0,0,"-"]}},objnames:{"0":["py","module","Python module"],"1":["py","class","Python class"],"2":["py","method","Python method"],"3":["py","function","Python function"]},objtypes:{"0":"py:module","1":"py:class","2":"py:method","3":"py:function"},terms:{"000":5,"000000":9,"000000000490046":12,"0052853855304419994":16,"006":5,"007476321":16,"0116lsst_hardware_i":16,"0116ynull00":16,"012":5,"0135lsst_hardware_i":16,"0135ynull00":16,"014":16,"014782":16,"017":16,"018":5,"019996643066430":16,"0200004577636730":16,"0230":16,"024":5,"03000068664550830":16,"0302lsst_hardware_z56":16,"0302znull00":16,"0316lsst_hardware_z":16,"0316znull00":16,"0331":16,"0396671":16,"03d4c13c0588":15,"043":16,"0441lsst_hardware_r36":16,"0441rnull00":16,"0451gnull00":16,"0451lsst_hardware_g":16,"0456lsst_hardware_z15":16,"0456znull00":16,"0500030517578130":16,"0514828":16,"0530":16,"05809":16,"0619inull00":16,"0619lsst_hardware_i45":16,"0621inull00":16,"0621lsst_hardware_i44":16,"0629":16,"063lsst_hardware_r44":16,"063rnull00":16,"0707518899702984":16,"0732":16,"0854431":16,"08it":17,"0974519855125831":16,"0ab":16,"0x7ff1eb4dcb80":17,"100":[16,17],"1000":[7,16],"10000":17,"101":6,"1024":[2,5],"106071":16,"10_000":12,"10_500":12,"11000":12,"111":16,"11159916":16,"1132895039613452":16,"115":16,"1153":16,"1155":17,"11999":5,"11_000":12,"12000":[5,16],"1208187":16,"123":[8,21],"1299972534179730":16,"133":16,"1330":16,"1399993896484430":16,"141":6,"1431":16,"1504382640457388e":16,"160":16,"16417071":16,"1763":16,"17768697380617842":16,"180d":16,"200":1,"2012":[15,16],"2013":15,"2014":[0,15],"2015":15,"2016":[15,16,18],"2017":[15,16,18],"2018":[15,16],"2020":2,"219286":16,"231":17,"234073":16,"23661":16,"244573":[2,5],"248458":16,"2529":16,"2530":16,"2629":16,"265":16,"2670":16,"2698":16,"270":16,"2700004577636730":16,"2730":16,"2790986958591953":16,"2906228005886078":16,"29346512":16,"295":[1,4,6],"2963623000791163":16,"3000":[5,12,16],"3025581":16,"304657e":5,"310000419616730":16,"3131":16,"315":17,"3199996948242230":16,"31st":9,"3230":16,"32702415":16,"33201916":16,"333333":9,"3378lsst_hardware_z":16,"3378znull00":16,"3392lsst_hardware_r8":16,"3392rnull00":16,"3398":16,"341871":16,"34999847412109430":16,"35000038146972730":16,"35291118":16,"3531":16,"3558lsst_hardware_u":16,"3558unull00":16,"3594inull00":16,"3594lsst_hardware_i47":16,"365315e":5,"365863e":5,"366418e":5,"366567e":5,"366673e":5,"3669lsst_hardware_z115":16,"3669znull00":16,"380535420823767":16,"3849inull00":16,"3849lsst_hardware_i":16,"3866lsst_hardware_r":16,"3866rnull00":16,"4000":17,"4036671":16,"4041lsst_hardware_z49":16,"4041znull00":16,"4048lsst_hardware_z":16,"4048znull00":16,"4096":22,"4141721":16,"4183713434799661":16,"4253lsst_hardware_r":16,"4253rnull00":16,"4270":16,"4623217":16,"4633":16,"46999931335449230":16,"4725":16,"4731":16,"47771573":16,"47it":17,"4809572":16,"489623964":16,"491012378302115":16,"4991132":16,"4998":16,"4999941":16,"4mm":[5,12],"5076399154905724":16,"5132":16,"518734756":16,"5298":16,"52it":17,"532":16,"546875":16,"5542":16,"5571732":16,"5824":16,"5842":16,"587281":16,"5899963378906230":16,"5931":16,"59823511":16,"60000038146972730":16,"61219":16,"61221":16,"61226":16,"61232":16,"61234":16,"61273":16,"6144":22,"61706":16,"61708":16,"61712":16,"61721":16,"61723":16,"6298":16,"631":16,"6327192":16,"6394937001578770":16,"640673882222792642":16,"6461":16,"652":16,"659192e":5,"6598":16,"6648107448125331":16,"666667":9,"672373506764416e":16,"67it":17,"6935133590087765":16,"7000007629394530":16,"701310611":16,"70999908447265630":16,"71307610":16,"7131":16,"7167320654263392":16,"730":16,"738421":16,"73904162":16,"7410398721694946":16,"7460306056323125":16,"7499537":[2,5],"751426e":5,"752069652":16,"7530":16,"7531":16,"7631":16,"766042":16,"7722551":16,"7872":16,"7915":16,"795492":16,"8000":[12,14],"811768":16,"81it":17,"8202672":16,"822":16,"8224":16,"839999914169311530":16,"8400":12,"8428":16,"845":17,"847191e":5,"87472":16,"879":17,"8830406166225832":16,"884":16,"8850":12,"8925748996019570":16,"8967":16,"9139623713549772":16,"9140921147995524":16,"91bg":22,"920":5,"9207042":16,"9235634":16,"93262":16,"933333":5,"93645":16,"939222":16,"940":5,"9430":16,"94894":16,"9598":16,"9599990844726630":16,"960":5,"960049e":5,"96096":16,"9630":16,"9798":16,"980":5,"981770815":16,"9830":16,"9898":16,"991539":16,"998845644":16,"99912220":16,"9998":16,"999999802004239":12,"\u03b1":6,"\u03b2":6,"abstract":2,"byte":7,"case":15,"class":[1,2,3,5,7,9,11,16],"const":[16,17],"default":[1,2,5,6,9,12,14,18,21],"export":20,"final":[14,15,20],"float":[1,2,4,5,6,9,14,16,17],"function":[2,6,7,8,9,10,11,12,15,17,21],"g\u00f6ttingen":12,"import":[1,2,3,5,7,8,9,10,12,14,15,16,17,21],"int":[1,2,3,4,5,6,7,9,12,17],"new":[2,8,20],"return":[1,2,3,4,5,6,7,8,9,10,12,14],"short":15,"static":[1,2,3,5,15,17],"true":[2,3,6,8,12,14,16,17],"try":[16,20],"while":[15,16],Axes:4,EBE:22,For:[9,11,16,17,20,21,22],GPS:[2,13,18],LOS:14,NOT:[9,12],Not:[19,20],The:[0,1,2,3,4,5,6,7,8,9,10,11,13,14,17,18,19,20,21,22],These:[13,19],USE:9,Useful:[1,18],__init__:[1,2,3,5,7,9],_end_mag:17,_filter:[8,21],_mag_arr:17,_no_atm:[8,21],_norm:12,_pwv_list:17,_pwv_model:15,_ref_mag:17,_start_mag:17,about:[16,22],abov:[3,12,15,16,17],abs:12,abs_mag:[1,4],abs_mb:1,absolut:[0,1,4],absorpt:[2,12,15,17],abstractvariablepwveffect:2,accept:2,access:[2,3,8,16,19,21],accessor:[9,21],accomplish:20,accord:15,account:17,accumul:16,accur:[1,2,9],achiev:16,across:[0,15,16,17],act:13,action:1,activ:20,actual:20,add:[1,2,7,12,15,16,17,20],add_effect:[2,17],add_pwv_columns_to_t:1,added:2,adding:[6,16],addit:[19,21],admir:15,advantag:19,affili:19,after:[1,2,17,18],against:[4,12,18],agn:22,airmass:[1,2,8,15,16,21],aitoff:16,all:[2,3,6,8,11,12,15,20,21],alloc:[1,7],allow:[12,21],almost:17,along:[2,14,16],alpha:[0,4,6,12,14,15,16,17],alreadi:[8,20],also:[6,7,10,13,15,17,19,20,21],alt:[2,5],alt_sch:[1,3,16],altern:[3,7,15],although:17,altitud:[0,2,5],amount:7,analysi:[4,11,13,21],analyz:21,angstrom:[2,4,5,8,12],ani:[6,9,15,17],answer:13,api:11,appar:6,append:[14,16],appli:[2,14],approach:[1,13,16,17],arang:[7,9,12,14,15,16,17],aren:10,arg:[2,12,14,15,16,17],arg_it:17,argument:[2,6,7,9,20],arrai:[2,4,5,6,7,8,9,14,16,17],ascens:[1,2,4,5],assert:9,assum:[1,2,8,9,11,15,18,19,21],assumed_pwv:[2,15],assumpt:11,astronom:[2,11,13,15,21,22],astronomi:11,astropi:[0,1,2,3,4,5,14,15,16,17],astyp:15,asynchron:1,atm:[8,17],atm_transmiss:2,atmospher:[1,2,4,5,6,8,9,11,12,13,14,15,17,21],atmospheric_continuum_fit:12,attribut:[2,9],auto:14,auto_built_model:15,automat:[9,10,21],avail:[2,3,5,9,11,13,15,16,19,20,21],averag:[2,5,12,15,16],average_norm_flux:5,avg:16,avoid:[14,15,20],awar:15,axes:[4,14,16,17],axi:[4,12,14,16],axis_row:16,axvlin:[14,16],axvspan:12,backward:2,band:[1,2,4,5,6,8,12,14,16,17,21],band_abbrev:14,band_data:[12,16],band_lett:12,band_nam:14,band_tot:12,bandpass:[4,12,14],bar:[3,6,17],base:[1,2,16,22],base_mag:14,base_s:14,baselin:[8,19,21],basi:[12,19],basic:[3,11],bb_temp:14,bbox_inch:17,becaus:7,becom:14,been:[7,17],begin:[9,15],behavior:[2,10,11],being:19,below:[0,2,8,11,12,13,15,16,17,18,19,20,21,22],best:[0,12,15],beta:[0,4,6,17],betoul:0,betoule_abs_mb:[0,16,17],betoule_alpha:0,betoule_beta:0,betoule_cosmo:[0,16,17],betoule_h0:0,betoule_omega_m:0,better:17,between:[8,15,17],bin:[2,12,15,16],binari:22,bit:14,black:13,black_body_s:14,blackbodi:14,blue:11,bluer:17,bodi:13,bool:[2,3,6,8,17],bound:[1,6,12,17,18],bound_c:18,bound_t0:18,bound_x0:18,bound_x1:18,bound_z:18,boundari:[9,15],box:11,branch:20,bright:14,brought:11,build:[2,18],built:[1,21],builtin:7,cach:11,cache_s:7,cadenc:[1,2,3,13,17,18,19],cadence_nam:16,cadence_sim:20,calc_airmass:[2,15,16],calc_calibration_factor_for_param:[6,17],calc_delta_bb_mag:14,calc_delta_mag:[6,17],calc_mu_for_model:6,calc_mu_for_param:[6,17],calc_transmiss:2,calcul:[2,4,6,9,12,14,16,17],calib_factor:17,calib_factor_with_ref:17,calibr:[1,5,6,11,17,18],calibrate_lc:[5,17],call:[1,7,9],callabl:7,camera:19,can:[1,2,3,5,7,9,11,12,15,16,17,19,20,21],capabl:21,care:16,cart:22,catalog:[1,5,17],ccd:[8,21],cell:[16,17],centigrad:18,cerro:[15,19],certain:22,chang:[2,4,5,6,11,12,14],character:19,check:[8,15,16,17],chi:1,child:1,chisq:1,choic:12,chosen:17,chunk:[3,16],clearli:16,clone:[19,20],close:2,cm2:5,cmap:4,code:[4,11,13,19,20],coef:14,col:16,collabor:11,collect:[2,5,6,9,21],color:[4,6,11,12,14,15,16],colorbar:16,column:[1,4,5,22],column_nam:1,com:[19,20],combin:[1,4,8,11,17,21],combined_data:16,combined_plasticc:16,combined_sncosmo:16,come:[21,22],command:20,commonli:21,compar:[4,12,15,16],compare_prop_effect:15,comparison:12,compat:[2,3,7,16],complet:[20,22],compon:[2,12,17],compos:[2,5],compress:20,concaten:[12,14],concentr:[2,5,12,14,15,16,17,18,19],concern:11,conda:20,conda_prefix:20,condit:[9,15],conduct:11,config:6,config_path:6,configur:[10,16,18],connector:1,consid:[11,15,17,19],consist:[4,17],constant:[4,6,15,16,17],construct:[2,3,7,17],contain:22,continu:[16,20],continuum:12,continuum_wavelength:12,contribut:[8,12,21],contributor:11,conveni:20,convert:[3,14],coolwarm:4,coordin:[4,16,22],copi:[1,5,9,15,17],corr_pwv_effect:17,corr_pwv_effects_with_ref_star:17,correct:[4,6,17],correct_mag:[6,17],corrected_delta_mag:17,corrected_delta_mag_with_ref:17,corrected_fiducial_mag:17,corrected_fiducial_mag_with_ref:17,corrected_mag:17,corrected_mag_with_ref:17,corrected_slop:17,corrected_slope_with_ref:17,correspond:[6,8,9,19,22],cosmo:[1,4,6,16,17],cosmolog:[0,1,4,6,16,17],could:17,count:3,count_light_curv:[3,16],coverag:15,creat:[1,2,5,9,12,14,15,17,20],create_cad:17,critic:4,cross:17,csv:[1,18],ctio:[15,16,18],ctio_weath:15,current:3,curv:[1,2,3,4,5,6,8,12,13,14,17,21,22],custom:[8,12,16,21],cut_pwv:18,cut_srfcpress:18,cut_srfcrh:18,cut_srfctemp:18,cut_zenithdelai:18,dai:[4,15],dark:[4,11,19],data:[1,2,3,4,5,6,7,9,11,13,15,18,20,21],data_cut:[15,16],datafram:[2,4,5],date:[2,9,15,16,22],datetim:[2,4,9,15],datetime_to_sec_in_year:9,deactiv:20,deal:9,dec:[1,2,4,5,16,22],decam:[8,19],decam_:[8,21],decam_atm:[8,21],decam_ccd:[8,21],decemb:9,decimal_digit:4,decl:[16,22],declin:[1,2,4,5],decompress:20,decor:7,def:[7,12,14,15,16,17],defin:[0,1,2,7,8,14,15,16,21],definit:[14,22],deg:[2,5,12,16],degre:[0,2,12],delai:18,delta:[4,6,14,17],delta_bb_mag:14,delta_fitted_color_with_ref:17,delta_fitted_corrected_color:17,delta_mag:[4,14,17],delta_mag_arr:4,delta_tabulated_corrected_color:17,demo:16,demo_header_fil:16,demo_model_with_pwv:16,demo_out_path:1,demo_seri:9,demonstr:[12,13,14,15,16,17],dens:17,densiti:[0,4,17],depend:[5,10,16,17,20],deprec:15,deprecationwarn:15,depth:[13,20],desc:[11,17,19],describ:[15,22],descript:[0,13,19,21,22],design:[2,11,21],desir:[2,20],destin:1,detail:[12,13,21],detect:22,detector:[8,12,21],determin:[0,2,5,6,8,14,15,17],develop:[11,13,19,22],deviat:2,dict:[1,2,4,6,16,17],dictionari:[2,4,6,7],differ:[2,4,5,15,16,17,18,21,22],difficulti:20,dilat:17,dimens:6,dimension:6,dimensionless:0,direct:17,directli:[2,6,9,10,15,20],directori:[1,3,18,20,22],disabl:[4,15,17],disagr:17,disk:[1,5,11],displai:3,distanc:[4,6],distinct:[11,22],distmod:17,distribut:[2,15,16],divid:[5,22],doc:[10,11,16],document:[8,11,13,19,21],doe:[15,17],doesn:16,don:[15,20],download:19,download_available_data:[15,16],dpi:16,draw:2,drawn:2,drop:2,drop_nondetect:2,dtype:[5,9],due:[2,6,17],duplic:[1,16],duplicate_plasticc_lc:1,duplicated_lc:16,dure:[9,10,17,19],each:[0,2,4,6,8,11,12,14,15,16,17,21,22],earli:19,earlier:17,earliest:9,eas:2,easier:17,easili:[4,16],echo:20,ecsv:[1,18],eff:14,eff_tick:14,eff_xlim:14,effect:[1,2,6,13],effect_fram:[2,16],effect_nam:[2,16],effort:[11,19],either:22,elaps:9,elimin:15,els:12,emphasi:11,end:[6,9,12,15],energi:[4,11,15,19],enforc:[1,15,17,18],enough:[12,15],ensur:[1,2,12,20],entir:[15,20],entri:7,enumer:12,env:20,env_var:20,environ:[3,10,16],environment:20,epoch:18,equal:2,equat:4,equidist:17,equinox:2,equival:[6,16],error:[2,15,17],establish:[2,16],estim:17,etc:20,evalu:[2,3,16,17],even:8,evenli:9,exactli:9,exampl:[11,13],exce:7,except:16,exclus:17,execut:11,exist:[1,8,20],exist_ok:[12,14,17],exit:[1,18,20],exp:14,expect:[2,3,4,13,16,17],expon:[4,14],extend:[4,9,10,16,17,18,21],extens:[1,18],extern:[10,19,21],extinct:22,extract:2,extrapol:15,extrem:17,factor:[6,12],factori:1,fail:20,fall:15,fals:[1,2,3,8,16],fanci:15,featur:[12,17],few:15,fid_pwv:14,fid_pwv_dict:6,fiduci:[5,6,8,13,14,17,21],fiducial_mag:6,fiducial_pwv:6,field:22,fig:[12,14,16,17],fig_dir:[12,14,17],figsiz:[4,12,14,15,16],figur:[4,12,15,16,17],file:[1,3,6,16,18,20],file_list:20,fill:[9,15],fill_between:12,filter:[13,21,22],filter_onli:12,filters_and_hardwar:12,find:[13,20],first:[2,3,9,16,22],fit:[0,1,4,6,12,16,22],fit_err:1,fit_func:12,fit_label:12,fit_lc:[1,6],fit_lsst_atm_continua:12,fit_mag:[6,17],fit_model:1,fit_param:[1,12],fit_pool_s:18,fit_results_input:1,fit_results_output:1,fit_sourc:18,fit_vari:18,fitlightcurv:1,fitresultstodisk:1,fitted_fiducial_mag:17,fitted_fiducial_param:17,fitted_mag:17,fitted_mag_with_ref:17,fitted_magnitud:17,fitted_mu:17,fitted_mu_with_ref:17,fitted_param:[4,17],fitted_paramet:17,fitted_params_with_ref:17,fitted_pwv_effect:17,fitting_cli:18,fitting_pool:1,fittingpipelin:[1,11],fix:[2,18],fixedrestransmiss:2,flag:20,flatlambdacdm:[1,4,6],float32:5,float64:9,float64str15float64float64float64str8:16,float64str2str12int32float32float32float32float32float32float32float32:16,floatorarrai:2,flt:22,fluctuat:17,flux:[1,2,4,5,14,16,17],flux_with_pwv:16,flux_without_atm:14,flux_without_pwv:16,fluxcal:22,fluxcalerr:22,fluxerr:16,focu:15,follow:[1,2,7,15,16,17,20,21,22],foo:7,forc:8,forget:20,form:[6,11],formal:22,format:[1,2,3,4,5,14,16,20],format_data_to_sncosmo:[3,16],formatted_lc:[3,16],fortran:3,fortun:15,frac:17,fraction:12,frame:[2,4],framealpha:[12,14,15,16],frequent:14,from:[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22],from_plasticc:[2,16],from_pwv_model:[2,15],from_suominet_receiv:[2,15,16],full:15,fulli:15,fundament:[13,17],futur:[15,20],g2_star:5,gain:[2,17],galact:16,gener:[4,16,17,19],get:[2,3,5,6,15,16,17],get_available_cad:[3,16],get_available_typ:5,get_bandpass:[8,12,14],get_config_pwv_v:[6,17],get_datafram:[5,12],get_model_head:[3,16],git:20,github:[11,19,20],given:[1,2,3,4,5,6,8,9,12,14,15,18,22],glass:[8,21],global:19,goe:12,goettingen:[5,19],good:15,gps_pwv:[15,16],gpsreceiv:[2,15,16],grei:[11,12,15],grid:16,grizi:17,group:21,gunzip:20,h17:16,hand:[19,22],handl:[3,10,15],hardwar:[8,12,21],has:[3,7,11,17],hashabl:7,have:[4,15,16,17,20],head:22,headach:20,header:[3,22],header_data:16,header_path:16,height:2,help:[1,2,8],here:[2,11,16,17,19],high:19,highest:2,highlight:4,hist:16,histogram:16,home:[6,20],horizon:16,host:[10,16,19,20],how:[2,10,11,12,13,15,17,21],howev:[2,11,17],http:[19,20],hubbl:[0,4],humid:[18,19],hundr:20,ibc:22,idea:15,ident:19,identifi:[3,22],iloc:9,ilot:22,imag:16,impact:13,implement:[5,7,21],impos:[1,7],impress:21,inch:4,includ:[1,2,4,5,6,11,13,16,17,19,20,21],include_pwv:1,incorpor:[9,16],increas:11,indefinit:20,independ:[2,11,17],index:[2,4,5,12,15,17,22],individu:[1,5,20],induc:14,inf:[1,18],inform:[9,10,11,16,21,22],initi:[1,2,13],input:[9,15],insert:[12,14,15,16,17],instanc:[1,2,5,11,15,18],instanti:[2,7,20],instead:[2,15,16,17],integr:10,intend:[9,22],interest:[10,13],intern:[15,19],interplai:17,interpol:[2,9,15],interpolated_pwv:15,interpolated_pwv_data:15,interv:19,interven:15,intrins:[0,4,11],involv:11,ipython:15,is_positive_airmass:16,issu:11,item:16,iter:[1,3,17],iter_lc:[3,16],iter_lcs_fixed_snr:17,iter_light_curve_iter_with_ref:17,iter_lim:[1,3,16,18],iter_tot:17,itertool:17,januari:9,job:15,jupyt:13,keep:15,kei:4,kelvin:14,kept:11,know:20,kwarg:[6,15],label:[4,12,14,15,16],labelpad:[12,16],lambda:15,larger:[4,16],largest:[12,17],lat:[2,5],later:[14,17,19],latest:9,latitud:[0,2,5],launch:11,lc_data:3,lc_data_set:16,lc_iter:3,lc_list:16,lc_tabl:5,least:15,leav:6,left:[12,14],left_ax:[12,14],left_twin_i:14,legaci:[8,19],legend:[12,14,15,16],len:[6,8,17,21],length:[5,16],lens:[8,12,21],less:[2,6],level:[7,9],librari:[5,12,19],light:[1,2,3,5,6,13,15,17,22],light_curv:[1,2,3,5,6,17],light_curve_fixed_snr:2,light_curves_input:1,light_curves_with_ref:17,lightgrei:16,like:[8,10,21],limit:[1,3,7,14,16],line:[1,2,4,14,16,20],linear:[9,14],linearli:[9,15],linestyl:[14,16],linewidth:[12,15],link:21,list:[1,2,3,4,5,6,14,16,17,19,20],load:[1,3,5,16],load_example_data:5,loadplasticcsim:1,loc:[12,14,15],local:3,locat:[2,16],log:14,lon:[2,5],longitud:[0,2,5],look:[10,16,17],los:14,los_tick:14,los_xlim:14,lower:[6,12,14,18],lru_cach:7,lsst:[8,13,14,15,16,18],lsst_:[8,21],lsst_atm:12,lsst_atm_norm:12,lsst_atmos_10:[8,21],lsst_atmos_std:[8,12,21],lsst_detector:[8,12,21],lsst_filter:12,lsst_filter_:[8,12,21],lsst_hardware_:[8,12,14,16,17,21],lsst_hardware_g:17,lsst_hardware_i:17,lsst_hardware_r:17,lsst_hardware_z:17,lsst_len:[8,21],lsst_lens:[8,12,21],lsst_m:[8,21],lsst_mirror:[8,12,21],lsst_total_:[8,12,21],lsst_total_u:8,lsst_u_band:8,lsstdesc:20,m_b:6,m_nu:[1,4,6],machin:3,made:[13,17],mag:[4,6,14,17],mag_dict:4,magnitud:[0,1,4,5,6,13,14,22],mai:[11,16],maintain:15,make:[15,17],makeup:11,manag:20,mani:16,manipul:[9,21],manual:[14,20],map:4,mask:1,master:20,match:1,mater:0,matplotlib:[4,12,14,15,16,17],matter:[4,15],max:[14,16],max_queu:1,max_siz:7,maximum:[1,7],maxwav:14,mdwarf:22,mean:[9,13,17],measur:[4,9,15,17,18],meet:11,memoiz:7,memori:[3,7,11,16],memorycach:7,messag:1,meta:[16,17,22],metadata_conflict:16,meteorolog:19,meter:[0,2],method:[2,7,9],microsecond:9,milki:22,millibar:18,millimet:18,mimic:16,min:[14,16],minim:[11,12,17],minimum:11,minut:19,minwav:14,mira:22,mirror:[8,12,21],miss:[1,2,4,9,15],mix:20,mjd:[2,5,15,16,22],mjd_val:15,mjdfltfieldphotflagphotprobfluxcalfluxcalerrpsf_sig1sky_sigzeroptsim_magob:16,mkdir:[12,14,17,20],mock:17,model:[1,3,4,5,6,11,12,13,14,16,17,19,21],model_for_sim:[1,16],model_with_pwv:[16,17],model_without_pwv:[16,17],modeled_tran:12,modifi:5,modul:[0,10,11,17,21],modular:11,moduli:[4,6],modulo:9,modulu:[4,6],moment:16,more:[2,9,10,11,13,14,16,17],most:10,motiv:17,mpc:[0,1,4,6],mu_pwv_effect:17,mu_pwv_effects_with_ref:17,multi:4,multipl:[1,4,16,17,18,19,22],must:[2,4,7],mwebv:22,name:[1,2,4,6,7,8,13,20,21,22],nan:[9,15],ndarrai:[2,4,5,6,9],ndof:1,nearbi:15,necessari:11,need:[8,15,16,19,20],neff:[1,4,6],negative_coord:16,nest:20,newli:8,next:[3,16,17],node:1,nois:[2,17],non:[12,16,17,22],none:[1,2,3,4,6,7,8,10,12,15,16],norm:14,norm_flux:5,norm_pwv:14,normal:[1,2,5,6,12,16,17,22],normalized_tran:12,notat:4,note:[12,17],notebook:[12,14,15,16,17],nuisanc:[0,4],num:4,num_lc:3,num_process:1,number:[1,3,4,9,11,18,21,22],numer:[2,3,17,18],numerical_error_offset:17,numerical_error_offset_ref:17,numpi:[7,9,12,14,15,16,17],numpy_arg:7,ob0:[1,4,6],object:[0,1,2,3,5,7,9,11,12,14,15,17,21,22],obs:[2,16,17],obs_tim:2,observ:[1,2,4,5,11,13,16,17,18,21,22],observatori:[0,2,15,19],observedcad:[1,2,16,17],offici:22,oldest:7,om0:[1,4,6],omega:4,onc:[1,8,20],one:[2,6,15,16,21],ones:12,ones_lik:12,onli:[8,16,18,19,21,22],onlin:13,open:[11,16],optic:[8,21],optim:[1,12],option:[1,2,3,4,6,7,18],orang:11,order:[1,4,7,9],origin:[16,19],other:[6,9,19,21],our:[11,12,15,16,17],out_dir:1,out_path:[1,18],outlin:[11,13,15,16,18,20,22],output:[1,18,20],over:[2,3,4,8,11,12,15,16,17,19,21],overal:[7,17],overlaid:4,overview:21,overwrit:[2,8],own:[12,16],packag:[0,2,3,8,9,10,11,13,16,19,20],page:[20,22],pair:22,panda:[5,9],pandas_obj:9,panel:4,paper:14,parallel:[1,11],param:[1,6,16,17],param_nam:16,paramet:[0,1,2,3,4,5,6,7,8,9,12,15,16,17,18],parent:[0,1,12,14,17,19,22],pars:15,part:20,parti:10,particular:[9,11],particularli:18,pass:[2,9,17],path:[1,3,6,12,14,15,16,17,18,19,20],pathlib:[12,14,17],pattern:2,pdf:[12,17],peak:16,peakmjd:16,per:[2,12,15,18],percentag:18,perform:[3,11,19,21],period:[9,15],periodic_interpol:[9,15],phase:[2,4,16],phenomena:[2,15],phot:[5,22],photflag:[2,22],photometr:22,photometri:[17,22],photprob:22,physic:[0,2,15,17],pick:15,pip:20,pipelin:[1,4,18],pipelineresult:1,pisn:22,plastic:1,plasticc:[1,2,20],plasticc_cad:16,plasticc_data_input:1,plasticc_lc:[3,16],pleas:[8,11,20],plot:[12,14,15,16,17],plot_airmass_valid:16,plot_cosmology_fit:4,plot_delta_bb_mag:14,plot_delta_color:[4,17],plot_delta_mag_vs_pwv:4,plot_delta_mag_vs_z:4,plot_delta_mu:[4,17],plot_derivative_mag_vs_z:4,plot_fitted_param:[4,17],plot_interpolation_valid:15,plot_lc:16,plot_lsst_atm:12,plot_lsst_filt:12,plot_magnitud:[4,17],plot_pwv_mag_effect:[4,17],plot_residuals_on_ski:4,plot_scale_factor:12,plot_snr_distribut:16,plot_spectral_templ:[4,17],plot_stellar_spectrum:12,plot_variable_pwv_sn_model:16,plot_year_pwv_vs_tim:[4,15],plt:[12,14,15,16,17],png:14,point:[2,16,18,22],polici:11,poly1d:12,polyfit:12,polynomi:12,pool:11,posit:19,positive_airmass:16,positive_coord:16,posixpath:6,possibl:11,postfix:22,practic:17,pre:[13,16,21],precipit:[2,13,14,15,19],precis:4,predefin:7,presenc:[6,17],pressur:[18,19],previou:17,primari:9,primarili:19,print:[3,5,7,9,12,15,16,17],prioriti:9,process:[1,11,15,17,18],produc:1,product:[1,17],profil:[1,8,18,19,21],progress:[3,6,17],project:[11,13,15,16,19,20,22],propag:[2,11,16],propagationeffect:2,properti:2,propog:2,provid:[1,2,3,4,6,7,11,12,13,18,19,21,22],psf_sig1:22,ptrobs_max:22,ptrobs_min:22,publish:19,pull:15,pwv:[1,2,4,5,6,12,13,16,20],pwv_arr:[4,6,17],pwv_data:15,pwv_eff:14,pwv_eff_label:14,pwv_effect:16,pwv_interpol:2,pwv_kpno:[2,12,14,15,16,19],pwv_label:14,pwv_lo:[2,14],pwv_los_label:14,pwv_mag:14,pwv_model:[2,5,15,16],pwv_model_year:18,pwv_seri:[2,4],pwv_val:[14,17],pwv_zenith:[2,15],pwvmodel:[2,5,15,16],pyplot:[12,14,15,16,17],python:[3,7,10,21],question:[11,13],queue:1,quick:[15,22],quickli:16,radian:16,rais:[2,11,15],raise_below_horizon:[2,16],random:2,rang:[4,12,15,16],rate:18,ratio:[2,12,17],rcparam:16,read:11,reader:11,real:[13,15],realist:[13,16],reason:[12,15],recalibr:5,receiv:2,receiver_id:18,recogn:2,record:[13,22],red:11,redder:17,reddest:12,redshift:[4,6,17],reduc:2,ref:17,ref_df:12,ref_pwv:[4,6],ref_star:18,refer:[1,4,5,6,11,14,18,19,22],reference_catalog:5,reference_idx:17,reference_light_curv:17,reference_pwv:17,reference_pwv_config:17,reference_pwv_v:17,reference_star:[12,17],referencecatalog:[5,17],referencestar:[5,12],reformat:16,region:12,regist:[8,9,10,21],register_decam_filt:8,register_lsst_filt:8,register_sncosmo_filt:8,registri:21,rel:[4,5,6,12,14,18,19],relat:[16,19],reli:12,remov:7,render:[13,16],replac:19,report:12,repositori:[16,19,20],repres:[0,1,2,4,5,11,12,15,19,22],represent:[1,4,5,15],reproduc:4,requir:[11,19,20,21],res:[12,14],resampl:[9,15],resample_data_across_year:[9,15],resampled_pwv:15,rescal:14,research:[11,19,21],resembl:2,reset_index:15,residu:4,resolut:[2,4,12,14,15,19],respect:17,respons:[3,6,8,10,12,19,21],restart:20,result:[1,2,4,8,11,21],retreiv:5,retri:20,retriev:[1,5,6,8,21],right:[1,2,4,5,12,15],right_ax:[12,14],right_twin_i:14,rizi:14,rom:15,rotat:[12,16],rough:12,roughli:14,routin:7,row:[1,4,22],rrlyra:22,rubin:[0,2,15],rudimentari:5,run:[1,3,11,16,18,20],run_async:1,runner:6,runtim:8,salt2:[1,2,4,16,17,18],same:[2,6,9,16,17],samp_wav:2,sampl:[2,14,16,17,19],saniti:17,satur:14,save:22,savefig:[12,14,17],scalar:2,scale:[12,14,15,16],scat:16,scatter:[2,15,16],sci_not:4,scienc:[11,19,21],scientif:[4,11,13,21],scipi:12,scratch:[16,20],script:8,sdssr:2,season:[2,15],seasonal_averag:2,seasonal_prop:15,seasonalpwvtran:[2,15],sec:[2,4],second:[9,16],section:[10,11],sed:[12,14,17],sed_with_pwv:14,see:[1,2,9,10,11,17,21],select:[13,15],self:[1,2,7],sensit:[8,17,21],separ:12,seri:[2,4,5,9,15],server:16,servic:19,set:[1,2,4,13,14,15,16,17,19,20],set_label:16,set_source_peakabsmag:[16,17],set_titl:[12,16],set_xlabel:[12,14,16],set_xlim:[12,14,16],set_xtick:14,set_xticklabel:14,set_ylabel:[12,14,16],set_ylim:[12,14],setup:1,setup_environ:10,sever:[11,13,20],shade:11,shape:[6,17],sharei:[12,14,16],sharex:[12,16],should:[2,3,4,7,17,22],show:[4,6,12,14,15,16,17],shown:[11,15,20],sight:[2,14,16],signal:[2,17],signatur:[1,2],signific:15,silent:16,sim_dir:[1,18],sim_magob:22,sim_model:1,sim_model_nam:22,sim_param:1,sim_peakmjd:16,sim_pool_s:18,sim_redshift_cmb:16,sim_salt2c:16,sim_salt2x0:16,sim_salt2x1:16,sim_sourc:18,sim_vari:18,similar:[2,9,16,17],similarli:16,simpl:15,simplic:17,simul:[1,2,3,4,5,6,13,14,17,20,22],simulate_lc:[2,16],simulate_lc_fixed_snr:[2,17],simulatelightcurv:1,simulation_input:1,simulation_output:1,simulation_pool:1,simulationtodisk:1,sinc:[3,16,17],singl:[1,2,4,9,11,15],size:[4,7,12,17],sky:22,skycoord:16,skynois:[2,17],slope:[4,6,17],slope_arr:4,slope_end:17,slope_end_idx:17,slope_start:17,slope_start_idx:17,slow:3,slsn:22,smooth:15,smoother:14,sn_coord:16,sn_magnitud:17,sn_model:1,snana:[3,16,22],snat_sim:[11,12,13,14,15,16,17,20,21],sncc:22,sncosmo:[1,2,3,4,5,6,8,12,14,16,17,18],sncosmo_lc:16,sne:[0,1,2,4,16,17,18],snia:22,snid:[1,22],snmodel:[1,2,6,16,17],snr:[2,16,17],softwar:[11,20],solstic:2,solut:7,some:[15,17],sometim:15,sort_index:15,sourc:[1,2,3,4,5,6,7,8,9,10,11,13,16,19],space:[8,19],span:15,spawn:18,specif:[12,13],specifi:[3,7,9,18,20],spectra:12,spectral:[2,4,5,12,15,16,18,19],spectral_templ:17,spectral_typ:[5,17],spectrophotometr:21,spectrum:[5,12,14],spectyp:12,spring:15,squar:1,srd:[8,19],stage:11,standard:[2,12,16,17],star:[1,5,18],start:[1,6,12,16,17,22],startup:20,stat:12,state:4,static_prop:15,staticpwvtran:[2,15,17],statist:[2,15],std:16,stdev:16,stellar:5,stellar_scale_factor:12,stellar_spectra:19,stellar_spectrum:12,step:[15,19,20],still:15,storag:20,store:[1,3],str:[1,2,3,4,5,6,8,12,17],stretch:[4,6,17],string:[1,4,22],subject:11,subplot:[4,12,14,16],subset:[9,16,20],subtract:5,subtyp:22,suffer:2,suit:20,summar:0,summari:[19,21,22],summed_residu:12,summer:15,suominet:[2,13,15,18,19],supernoca:5,supernova:[1,2,4,6,11,13,15,16,18,19,21,22],supernova_model:2,supp_year:[2,9],supplement:[2,9,15],supplementari:9,supplemented_data:[9,15],supplemented_pwv:15,support:[2,5,7,9,11,21],suptitl:17,sure:17,surfac:18,survei:[8,11,15,19],svo:19,synchron:1,synthet:[17,19],sys:[12,14,15,16,17],system:[2,21],tabl:[0,1,2,3,5,6,16,17,22],tabul:[4,6,16,17],tabular:21,tabulate_fiducial_mag:[6,17],tabulate_mag:[6,17],tabulated_delta_mag:17,tabulated_fiducial_mag:17,tabulated_mag:17,tabulated_magnitud:17,tabulated_pwv_effect:17,tabulated_slop:17,take:[12,16,17,19,20],taken:[4,9,13,15,19,22],tar:20,target:[2,5,22],task:9,tcmb0:[1,4,6],tde:22,team:19,technic:[11,21],techniqu:11,telolo:[15,19],temp:14,temperatur:[14,18,19],templat:[2,4,16,18],tempor:[15,16,17],ten:15,test:[17,20],text:17,than:[2,4,6],thei:[8,9,15],them:20,theoret:13,thi:[2,6,9,10,11,12,13,14,15,16,17,18,19,20,21,22],thing:16,think:[15,17],third:10,those:13,three:[15,16],through:[2,9,11,15,17],throughput:[4,8,19,21],thu:3,tight:17,tight_layout:[14,16],time:[2,4,5,8,9,11,13,15,16,17,19],time_format:[2,5,15],timebandfluxfluxerrzpzpsi:16,timedelta:15,timeout:20,timezon:15,titl:15,to_csv:1,to_list:1,to_panda:[5,12],to_sncosmo:2,togeth:11,tol:12,too:15,tool:21,total:[8,12,17,18,21],touch:20,tqdm:17,tran:[8,12],transit:15,transmiss:[2,4,8,12,14,16,17],transmission_r:2,trapz:14,treat:7,trend:17,tri:20,trigger:22,tsu:[9,15,21],tsuaccessor:9,tupl:[1,2,4,6,12],twin_ax:12,twini:14,twinx:12,two:15,txt:20,type:[1,2,3,4,5,6,8,9,10,11,12,13,16,17,19,22],ugrizi:[8,12,16,21],ulen:22,ultim:11,unchang:6,uncorrect:17,under:9,underli:[2,5,10,15],understand:[11,17],unfortun:15,uniform:17,uniformli:15,union:[1,2,4,5,6,9],uniqu:22,unit:[0,2,4,14,16,18],unless:10,unlik:[2,17],unset:20,updat:16,upper:[6,12,14,15,18],url:20,usag:[13,18],use:[1,2,3,4,5,6,14,15,16,17,18,22],used:[0,1,2,3,4,5,6,9,14,15,16,18,19,20,21,22],useful:18,user:[15,21],using:[1,2,3,5,9,13,15,16,19,20,22],utc:15,util:[5,8,20],v1_transmiss:[2,12,14],val:[6,17],valid:[15,16,19],valu:[0,1,2,4,5,6,8,9,14,15,16,17,18,19,22],valueerror:2,vapor:[2,13,14,15,19],vari:[1,2,6,15,18],variabl:[2,12,13,15,16,18,19,20],variable_prop:15,variablecatalog:[1,5],variablepropagationeffect:2,variablepwvtran:[2,15,16],variat:[2,4,17],varieti:19,variou:[2,19,20,21],vera:0,verbos:[3,6,16,17,20],verifi:16,version:[13,17],via:[3,6,20,21],visual:[4,16,17],vmax:16,vmin:16,vparam:[1,6,17,18],vro:2,vro_altitud:0,vro_latitud:0,vro_longitud:0,vstack:16,wai:22,want:20,water:[2,13,14,15,19],wave:[2,8,12,14,16],wave_arr:4,wave_rang:12,wavelength:[2,4,5,8,12,14,16],weather:[9,13,15],weather_data:15,weather_for_year:15,websit:19,welcom:11,well:11,were:[3,12,13,19,22],wfirst:16,wget:20,what:[9,16,17],when:[1,2,6,7,8,9,16,17,18],where:[20,22],wherev:11,whether:[2,8],which:[1,2,5,15,16,18],window:4,winter:[2,15],within:[8,9,11,18],without:[2,8,16,21],work:[3,6,11,16,17,20],worker:18,world:[13,15],would:9,wrap_at:16,wrapper:7,wrestl:14,write:[1,18],written:[11,13,16],x_1:6,x_arr:7,x_val:15,xlabel:15,xlim:15,xvzf:20,y_arr:7,yaml:6,year:[2,4,8,9,15,16,21],yield:[3,17],ylabel:15,ylim:[14,15],ymin:14,you:[2,3,10,15,20],your:16,z_arr:[4,6,17],z_val:17,zenith:[2,5,15,16,18,19],zenodo:[19,20],zero:[2,17,22],zeropt:22,zip:[16,17],zorder:12,zpsy:[2,17]},titles:["snat_sim.constants","snat_sim.fitting_pipeline","snat_sim.models","snat_sim.plasticc","snat_sim.plotting","snat_sim.reference_stars","snat_sim.sn_magnitudes","snat_sim.utils.caching","snat_sim.filters","snat_sim.utils.time_series","snat_sim.utils","Impact of Chromatic Effects on LSST SNe Ia","LSST Filters","Notebook Summaries","Effective PWV on a Black Body","PWV Modeling","Simulating Light-Curves with a Cadence","PWV effects on SN Magnitude","Command Line Interface","Data Provenance","Installation and Setup","Package Integrations","PLaSTICC Data Model"],titleterms:{"function":4,Adding:16,The:[12,15,16],With:17,api:7,appar:17,approach:15,argument:18,atm:12,atmospher:[16,19],autom:15,baselin:12,black:14,bodi:14,build:15,cach:7,cadenc:16,camera:21,chang:17,chromat:11,color:17,command:18,configur:20,constant:0,contribut:11,ctio:19,curv:[11,16,18,19,20],dark:21,data:[16,19,22],decam:21,distanc:17,doc:[1,2,3,4,5,6,8,9],download:20,effect:[11,14,15,16,17],energi:21,environ:20,exampl:[1,2,3,5,7,8,9],fiduci:12,file:22,filter:[8,12,19],fit:[17,18],fitting_pipelin:1,flux:12,format:22,impact:[11,17],instal:20,integr:21,interfac:18,legaci:21,light:[11,16,18,19,20],line:18,lsst:[11,12,19,21],magnitud:17,measur:19,model:[2,15,18,22],modul:[1,2,3,4,5,6,7,8,9],modulu:17,name:18,notebook:13,organiz:22,overview:11,packag:21,panda:21,pipelin:11,plasticc:[3,16,22],plot:4,propag:15,proven:19,pwv:[14,15,17,18],refer:17,reference_star:5,rel:17,result:17,seri:21,set:12,setup:20,sim:20,simul:[11,16,18,19],sn_magnitud:6,snat_sim:[0,1,2,3,4,5,6,7,8,9,10],sncosmo:21,sne:11,sourc:20,space:21,spectra:19,spectral:17,star:17,stellar:[12,19],submodul:10,summari:[2,4,13],survei:21,templat:17,time:21,time_seri:9,usag:[1,2,3,5,7,8,9,11],using:17,util:[7,9,10,21],without:17,your:20}})
Search.setIndex({docnames:["api/constants","api/filters","api/fitting_pipeline","api/lc_simulation","api/models","api/plasticc","api/plotting","api/reference_stars","api/sn_magnitudes","api/utils_caching","api/utils_time_series","index","notebooks/lsst_filters","notebooks/notebook_summaries","notebooks/pwv_eff_on_black_body","notebooks/pwv_modeling","notebooks/simulating_lc_for_cadence","notebooks/sne_delta_mag","overview/command_line","overview/data_provenance","overview/install","overview/plasticc_model"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":3,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":2,"sphinx.domains.rst":2,"sphinx.domains.std":1,"sphinx.ext.viewcode":1,nbsphinx:3,sphinx:56},filenames:["api/constants.rst","api/filters.rst","api/fitting_pipeline.rst","api/lc_simulation.rst","api/models.rst","api/plasticc.rst","api/plotting.rst","api/reference_stars.rst","api/sn_magnitudes.rst","api/utils_caching.rst","api/utils_time_series.rst","index.rst","notebooks/lsst_filters.nblink","notebooks/notebook_summaries.rst","notebooks/pwv_eff_on_black_body.nblink","notebooks/pwv_modeling.nblink","notebooks/simulating_lc_for_cadence.nblink","notebooks/sne_delta_mag.nblink","overview/command_line.rst","overview/data_provenance.rst","overview/install.rst","overview/plasticc_model.rst"],objects:{"snat_sim.filters":{register_decam_filters:[1,1,1,""],register_lsst_filters:[1,1,1,""],register_sncosmo_filter:[1,1,1,""]},"snat_sim.fitting_pipeline":{FittingPipeline:[2,2,1,""],KillSignal:[2,2,1,""],OutputDataModel:[2,2,1,""],ProcessManager:[2,2,1,""]},"snat_sim.fitting_pipeline.FittingPipeline":{__init__:[2,3,1,""]},"snat_sim.fitting_pipeline.OutputDataModel":{__init__:[2,3,1,""],build_masked_entry:[2,3,1,""],build_table_entry:[2,3,1,""],column_names:[2,3,1,""]},"snat_sim.fitting_pipeline.ProcessManager":{__init__:[2,3,1,""],kill:[2,3,1,""],run:[2,3,1,""],run_async:[2,3,1,""],wait_for_exit:[2,3,1,""]},"snat_sim.lc_simulation":{calc_x0_for_z:[3,1,1,""],create_observations_table:[3,1,1,""],iter_lcs_fixed_snr:[3,1,1,""],simulate_lc:[3,1,1,""],simulate_lc_fixed_snr:[3,1,1,""]},"snat_sim.models":{FixedResTransmission:[4,2,1,""],PWVModel:[4,2,1,""],SNModel:[4,2,1,""],SeasonalPWVTrans:[4,2,1,""],StaticPWVTrans:[4,2,1,""],VariablePWVTrans:[4,2,1,""],VariablePropagationEffect:[4,2,1,""]},"snat_sim.models.FixedResTransmission":{__init__:[4,3,1,""],calc_transmission:[4,3,1,""]},"snat_sim.models.PWVModel":{__init__:[4,3,1,""],calc_airmass:[4,3,1,""],from_suominet_receiver:[4,3,1,""],pwv_los:[4,3,1,""],pwv_zenith:[4,3,1,""],seasonal_averages:[4,3,1,""]},"snat_sim.models.SeasonalPWVTrans":{assumed_pwv:[4,3,1,""],propagate:[4,3,1,""]},"snat_sim.models.StaticPWVTrans":{__init__:[4,3,1,""],propagate:[4,3,1,""],transmission_res:[4,3,1,""]},"snat_sim.models.VariablePWVTrans":{__init__:[4,3,1,""],assumed_pwv:[4,3,1,""],propagate:[4,3,1,""]},"snat_sim.models.VariablePropagationEffect":{propagate:[4,3,1,""]},"snat_sim.plasticc":{count_light_curves:[5,1,1,""],duplicate_plasticc_sncosmo:[5,1,1,""],extract_cadence_data:[5,1,1,""],format_plasticc_sncosmo:[5,1,1,""],get_available_cadences:[5,1,1,""],get_model_headers:[5,1,1,""],iter_lc_for_cadence_model:[5,1,1,""],iter_lc_for_header:[5,1,1,""]},"snat_sim.plotting":{plot_cosmology_fit:[6,1,1,""],plot_delta_colors:[6,1,1,""],plot_delta_mag_vs_pwv:[6,1,1,""],plot_delta_mag_vs_z:[6,1,1,""],plot_delta_mu:[6,1,1,""],plot_delta_x0:[6,1,1,""],plot_derivative_mag_vs_z:[6,1,1,""],plot_fitted_params:[6,1,1,""],plot_magnitude:[6,1,1,""],plot_pwv_mag_effects:[6,1,1,""],plot_residuals_on_sky:[6,1,1,""],plot_spectral_template:[6,1,1,""],plot_year_pwv_vs_time:[6,1,1,""],sci_notation:[6,1,1,""]},"snat_sim.reference_stars":{average_norm_flux:[7,1,1,""],divide_ref_from_lc:[7,1,1,""],get_available_types:[7,1,1,""],get_ref_star_dataframe:[7,1,1,""],get_stellar_spectra:[7,1,1,""],interp_norm_flux:[7,1,1,""]},"snat_sim.sn_magnitudes":{calc_calibration_factor_for_params:[8,1,1,""],calc_delta_mag:[8,1,1,""],calc_mu_for_model:[8,1,1,""],calc_mu_for_params:[8,1,1,""],correct_mag:[8,1,1,""],fit_fiducial_mag:[8,1,1,""],fit_mag:[8,1,1,""],get_config_pwv_vals:[8,1,1,""],tabulate_fiducial_mag:[8,1,1,""],tabulate_mag:[8,1,1,""]},"snat_sim.utils":{caching:[9,0,0,"-"],time_series:[10,0,0,"-"]},"snat_sim.utils.caching":{MemoryCache:[9,2,1,""],numpy_cache:[9,1,1,""]},"snat_sim.utils.caching.MemoryCache":{__init__:[9,3,1,""]},"snat_sim.utils.time_series":{TSUAccessor:[10,2,1,""],datetime_to_sec_in_year:[10,1,1,""]},"snat_sim.utils.time_series.TSUAccessor":{__init__:[10,3,1,""],periodic_interpolation:[10,3,1,""],resample_data_across_year:[10,3,1,""],supplemented_data:[10,3,1,""]},snat_sim:{constants:[0,0,0,"-"],filters:[1,0,0,"-"],fitting_pipeline:[2,0,0,"-"],lc_simulation:[3,0,0,"-"],models:[4,0,0,"-"],plasticc:[5,0,0,"-"],plotting:[6,0,0,"-"],reference_stars:[7,0,0,"-"],sn_magnitudes:[8,0,0,"-"]}},objnames:{"0":["py","module","Python module"],"1":["py","function","Python function"],"2":["py","class","Python class"],"3":["py","method","Python method"]},objtypes:{"0":"py:module","1":"py:function","2":"py:class","3":"py:method"},terms:{"000000":10,"000000000490046":12,"010":16,"0115unull00":16,"0124":16,"0129":16,"014353":16,"015":16,"02218671888113022":16,"0223":16,"0229":16,"0298":16,"0305gnull00":16,"035":16,"0398":16,"03d4c13c0588":15,"040213":16,"05226231":16,"0616961":16,"0633261":16,"0642":16,"077633":16,"089334056006399e":16,"0ab":17,"0lsst_hardware_g25":17,"0lsst_hardware_i25":17,"0lsst_hardware_r25":17,"0lsst_hardware_y25":17,"0lsst_hardware_z25":17,"100":[3,16,17],"1000":9,"10000":17,"101":8,"1024":4,"109375":16,"10_000":[12,16],"10_500":12,"11000":12,"111":16,"1153951":16,"1155":17,"11_000":12,"12000":16,"12109375":16,"12165431":16,"123":1,"137":16,"141":8,"14511533":16,"16801":15,"17167":15,"17532":15,"180d":16,"1998":16,"2012":[15,16],"2013":15,"2014":[0,15],"2015":15,"2016":[15,16,18],"2017":[15,16,18],"2018":[15,16],"2020":4,"205":16,"20it":17,"2131":16,"22867831":16,"2299inull00":16,"231":17,"2440067":16,"244573":4,"2698":16,"270":16,"2831":16,"28750507":16,"2879rnull00":16,"293896":16,"295":[3,5,6,8],"3000":[12,16],"3097inull00":16,"315":17,"31st":10,"3293rnull00":16,"32it":17,"33135343":16,"3314":16,"3328unull00":16,"333333":10,"3355znull00":16,"3365rnull00":16,"3456":16,"3507ynull00":16,"3514ynull00":16,"3544ynull00":16,"355":17,"3731":16,"3823":16,"392":16,"393404332":16,"4000":17,"4032197":16,"4096":21,"4133":16,"4177971":16,"4430":16,"4433":16,"447628":16,"4507362":16,"4630":16,"46613329":16,"4707477":16,"4737479":16,"4764":16,"47it":17,"4855":16,"4898":16,"4931":16,"4998":16,"4mm":[7,12],"50114446":16,"531":16,"5330":16,"5424":16,"54368527":16,"558":17,"5598":16,"57576321":16,"5851":16,"58881712":16,"59528974":16,"598":16,"61387":16,"61389":16,"61390":16,"61392":16,"61394":16,"61405":16,"6144":21,"61480":16,"615271":16,"617":17,"617575":16,"61875":16,"61878":16,"61881":16,"61883":16,"61885":16,"6282458305358887":16,"628982424037531e":16,"642":16,"642986":16,"6504642":16,"659918":16,"66339093":16,"666667":10,"6693171":16,"67514372":16,"684435229":16,"6925542":16,"6935":16,"6947411":16,"7004971":16,"70918":16,"7130":16,"7231":16,"7251":16,"731":16,"731249":16,"7330":16,"7469527125358582":16,"7499537":4,"750":16,"752069652":16,"76683334":16,"7930":16,"7931":16,"8000":[12,14],"8032235":16,"814":16,"8170":16,"8230":16,"8230511":16,"831097":16,"8313211":16,"8400":12,"8534":16,"8628":16,"86it":17,"87183":16,"8831":16,"88390064":16,"8850":12,"8911362":16,"89297220":16,"89714771":16,"9143":16,"91bg":21,"930":16,"93107":16,"9428":16,"9430":16,"9437741":16,"94777":16,"9498":16,"95182":16,"9530":16,"9531":16,"95it":17,"96006":16,"96628":16,"9702znull00":16,"9734ynull00":16,"9747ynull00":16,"978ynull00":16,"9831":16,"9871rnull00":16,"9872gnull00":16,"992rnull00":16,"9958znull00":16,"999999802004232":12,"\u03b1":8,"\u03b2":8,"abstract":4,"byte":9,"class":[2,4,9,10,16],"default":[3,4,5,7,8,10,12,14,18],"export":20,"final":[14,15,20],"float":[2,3,4,5,6,7,8,10,14,16],"function":[1,2,4,8,9,10,11,12,15,17],"g\u00f6ttingen":12,"import":[1,2,3,4,5,9,10,12,14,15,16,17],"int":[2,3,4,5,6,7,8,9,10,12,16],"new":[1,20],"return":[1,2,3,4,5,6,7,8,9,10,12,14],"short":15,"static":[4,15,17],"true":[1,3,5,8,12,14,16,17],"try":[2,16,20],"while":[11,15],Axes:6,EBE:21,For:[10,11,16,17,20,21],GPS:[4,13,18],LOS:14,NOT:[10,12],Not:[19,20],The:[0,1,2,3,4,5,6,7,8,9,10,11,13,14,17,18,19,20,21],These:[11,13,19],USE:10,Useful:[2,18],__init__:[2,4,9,10],_end_mag:17,_filter:1,_mag_arr:17,_no_atm:1,_norm:12,_pwv_list:17,_pwv_model:15,_ref_mag:17,_sourc:17,_start_mag:17,about:[16,21],abov:[5,12,15,16,17],abs:12,abs_mag:[3,6],absolut:[0,3,6],absorpt:[4,12,17],accept:4,access:[1,4,5,19],accessor:10,accomplish:20,accor:15,accord:5,account:17,accumul:16,accur:10,achiev:16,across:[0,15,16,17],act:[4,13],activ:20,actual:20,add:[3,5,9,12,16,17,20],add_effect:[4,17],added:[3,4],adding:[8,16],addit:19,admir:15,advantag:19,affili:19,after:[4,17,18],against:[6,12,18],agn:21,airmass:[1,4,15,16],aitoff:16,all:[1,2,5,8,11,12,15,20],alloc:[2,9],allow:12,almost:17,along:[4,14,16],alpha:[0,6,8,12,14,15,16,17],alreadi:[1,20],also:[5,8,9,13,17,19,20],alt:4,alt_sch:[2,5,16],alt_sched_rol:16,altern:[5,9,15],although:17,altitud:[0,4],amount:9,analysi:[6,11,13,19],angstrom:[1,4,6,7,12],ani:[3,8,10,15,17],answer:13,api:[11,19],appar:8,append:[14,16],appli:14,approach:[2,3,13,16,17],arang:[9,10,12,14,15,16,17],arg:[4,12,14,15,16],argument:[4,8,9,10,20],arrai:[1,3,4,5,6,7,8,9,10,14,16],ascens:[4,6],assert:10,assum:[1,4,10,11,15,18,19],assumed_pwv:[4,15],assumpt:11,astronom:[11,13,15,21],astronomi:11,astropi:[0,3,4,5,6,7,14,15,16,17],astyp:15,asynchron:2,atm:[1,17],atm_transmiss:4,atmospher:[1,2,4,6,7,8,10,11,12,13,14,15,17],atmospheric_continuum_fit:12,attribut:[4,10],auto:14,auto_built_model:15,automat:10,avail:[4,5,7,10,11,13,15,16,19,20],averag:[4,7,12,15,16],average_norm_flux:7,avg:16,avoid:[14,15,20],awar:[12,15],axes:[6,14,16,17],axi:[6,12,14,16],axis_row:16,axvlin:[14,16],axvspan:12,backward:4,band:[1,3,6,7,8,12,14,16,17],band_abbrev:14,band_data:[12,16],band_lett:12,band_nam:14,band_pass:3,band_tot:12,bandpass:[3,6,12,14],bar:[3,5,8],base:[4,16,21],base_mag:14,base_s:14,baselin:[1,19],basi:[12,19],basic:[5,11],bb_temp:14,bbox_inch:17,becaus:9,becom:14,been:[9,17],befor:2,begin:[10,15],behavior:4,being:19,below:[0,1,3,4,11,12,13,15,16,17,18,19,20,21],best:[0,12,15],beta:[0,6,8,17],betoul:0,betoule_abs_mb:0,betoule_alpha:0,betoule_beta:0,betoule_cosmo:[0,17],betoule_h0:0,betoule_omega_m:0,better:17,between:[1,15,17],bin:[4,12,15,16],binari:21,bit:14,black:13,black_body_s:14,blackbodi:14,block:2,blue:11,bluer:17,bodi:13,bool:[1,3,5,8],both:3,bound:[2,8,12,17,18],bound_c:18,bound_t0:18,bound_x0:18,bound_x1:18,bound_z:18,boundari:[10,15],box:11,branch:20,bright:[11,14],build:[4,18],build_masked_entri:2,build_table_entri:2,built:2,builtin:9,bulk:11,cach:11,cache_s:9,cadenc:[2,3,5,13,17,18,19],cadence_sim:[16,20],calc_airmass:[4,15,16],calc_calibration_factor_for_param:[8,17],calc_delta_bb_mag:14,calc_delta_mag:[8,17],calc_mu_for_model:8,calc_mu_for_param:[8,17],calc_transmiss:4,calc_x0_for_z:3,calcul:[4,6,8,10,12,14,16,17],calib_factor:17,calib_factor_with_ref:17,calibr:[2,7,8,11,17,18],call:[2,9,10],callabl:[2,9],camera:19,can:[2,3,4,5,9,10,11,12,15,16,17,19,20],care:16,cart:21,ccd:1,cell:17,centigrad:18,cerro:[15,19],certain:21,chang:[6,7,8,12,14],character:19,check:[1,5,15,16,17],child:2,choic:12,chosen:[3,17],chunk:5,clearli:16,clone:[19,20],close:4,cm2:7,cmap:6,code:[6,11,13,19,20],coef:14,coher:2,col:16,collabor:11,collect:[2,3,4,7,8,10],color:[6,8,11,12,14,15,16],colorbar:16,column:[2,3,6,7,21],column_nam:2,com:[19,20],combin:[1,2,3,6,11],combined_data:16,combined_plasticc:16,combined_sncosmo:16,come:[3,21],command:20,compar:[6,12,15,16],compare_prop_effect:15,comparison:12,compat:[4,5,9,16],complet:[20,21],complex:11,compon:[12,17],compos:4,compress:20,concaten:[12,14],concentr:[2,4,12,14,15,16,17,18,19],concern:11,conclus:11,conda:20,conda_prefix:20,condit:[10,15],conduct:11,config:8,config_path:8,configur:18,consid:[15,17,19],consist:[6,17],constant:[6,8,15,16,17],construct:[4,5,9,17],contain:21,continu:[2,16,20],continuum:12,continuum_wavelength:12,contribut:[1,12],contributor:11,conveni:20,convent:3,convert:[5,14],coolwarm:6,coordin:[6,16,21],copi:[7,10,15],core:1,corr_pwv_effect:17,corr_pwv_effects_with_ref_star:17,correct:[6,8,17],correct_mag:[8,17],corrected_delta_mag:17,corrected_delta_mag_with_ref:17,corrected_fiducial_mag:17,corrected_fiducial_mag_with_ref:17,corrected_mag:17,corrected_mag_with_ref:17,corrected_slop:17,corrected_slope_with_ref:17,correspond:[1,6,8,10,19,21],cosmo:[3,5,6,8],cosmolog:[0,3,5,6,8,16,17],could:17,count:5,count_light_curv:[5,16],coverag:15,creat:[2,3,4,10,12,14,15,17,20],create_observations_t:[3,17],critic:6,cross:17,csv:[2,18],ctio:[15,16,18],ctio_weath:15,current:5,curv:[1,2,3,4,5,6,7,8,12,13,14,17,21],custom:[1,12,16],cut_press:18,cut_pwv:18,cut_rh:18,cut_temp:18,cut_zenith_delai:18,dai:[6,15],dark:[6,11,19],data:[2,4,5,6,9,10,11,13,15,18,20],data_cut:[15,16],datafram:[4,6,7],date:[3,4,10,15,16,21],datetim:[4,6,10,15],datetime_to_sec_in_year:10,deactiv:20,deal:10,dec:[4,6,16,21],decam:[1,19],decam_:1,decam_atm:1,decam_ccd:1,decam_g:3,decam_i:3,decam_r:3,decam_z:3,decemb:10,decimal_digit:6,decl:[16,21],declin:[4,6],decompress:20,decor:9,def:[9,12,14,15,16,17],defin:[0,1,2,3,4,9,14,15,16],definit:[14,21],deg:[4,12,16],degre:[0,4,12],delai:18,delta:[6,8,14,17],delta_bb_mag:14,delta_fitted_color_with_ref:17,delta_fitted_corrected_color:17,delta_mag:[6,14,17],delta_mag_arr:6,delta_tabulated_corrected_color:17,demo_cad:16,demo_cadence_header_fil:16,demo_header_path:16,demo_model_with_pwv:16,demo_out_path:2,demo_seri:10,demonstr:[2,3,12,13,14,16,17],dens:17,densiti:[0,6,17],depend:[1,16,17,20],deprec:15,deprecationwarn:15,depth:[13,20],desc:[11,19],describ:[15,21],descript:[0,13,19,21],design:[4,11],desir:20,detail:[12,13],detect:21,detector:[1,12],determin:[0,1,3,4,7,8,14,17],develop:[11,13,19,21],deviat:3,dict:[2,3,4,6,8,16,17],dictionari:[4,6,8,9],differ:[4,6,15,16,17,18,21],difficulti:20,dilat:17,dimens:8,dimension:8,dimensionless:0,direct:17,directli:[8,10,15,20],directori:[5,20,21],disabl:[6,15],disagr:17,disk:[7,11],displai:5,distanc:[6,8],distinct:[11,21],distmod:17,distribut:[3,16],divid:[7,21],divide_ref_from_lc:[7,17],doc:11,document:[1,11,13,19],doe:[15,17],doesn:16,don:[15,20],download:19,download_available_data:[15,16],dpi:16,draw:[4,11],drawn:3,drop:5,drop_nondetect:5,dtype:10,due:[4,8,17],duplic:16,duplicate_plasticc_sncosmo:[5,16],duplicated_lc:16,dure:[5,10,16,17,19],each:[0,1,3,4,5,6,8,11,12,14,15,16,17,21],earli:19,earlier:17,earliest:10,eas:4,easi:5,easier:17,easili:[6,16],echo:20,eff:14,eff_tick:14,eff_xlim:14,effect:[2,4,8,13],effect_fram:[4,16],effect_nam:[4,16],effort:[11,19],either:21,elaps:10,elimin:15,els:12,emphasi:11,end:[8,10,12,15],energi:[6,11,19],enforc:[2,15,17,18],enough:[12,15],ensur:[4,12,20],entir:[15,20],entri:[2,9],enumer:12,env:20,env_var:20,environ:[5,16],environment:20,epoch:18,equal:3,equat:6,equidist:17,equinox:4,equival:[8,16],error:[3,15,17],establish:16,estim:17,etc:20,evalu:[3,4,5,16,17],even:1,evenli:10,exactli:10,exampl:[11,13],exce:9,except:[2,16],exclus:17,execut:[2,11],exist:[1,10,20],exist_ok:[12,14,17],exit:[2,18,20],exp:14,expect:[3,4,5,6,13,16,17],expon:[6,14],extend:[6,10,16,17,18],extens:[2,18],extern:19,extinct:21,extract:5,extract_cadence_data:5,extrapol:15,extrem:17,factor:[8,12],fail:[2,20],fall:15,fals:[1,2,5,16],fanci:15,featur:[12,17],feed:3,few:15,fid_pwv:14,fid_pwv_dict:8,fiduci:[1,7,8,13,14,17],fiducial_mag:8,fiducial_pwv:8,field:21,fig:[12,14,16,17],fig_dir:[12,14,17],figsiz:[6,12,14,15,16],figur:[6,12,15,16,17],file:[2,5,8,16,18,20],file_list:20,fill:[10,15],fill_between:12,filter:[5,13,14,16,17,21],filter_onli:12,filters_and_hardwar:12,find:[5,13,20],finish:2,first:[5,10,11,16,21],fit:[0,2,5,6,8,12,16,21],fit_fiducial_mag:[8,17],fit_func:12,fit_label:12,fit_lc:8,fit_lsst_atm_continua:12,fit_mag:[8,17],fit_model:[2,8],fit_param:12,fit_pool_s:18,fit_sourc:18,fit_vari:18,fitted_fiducial_mag:17,fitted_fiducial_param:17,fitted_mag:17,fitted_mag_with_ref:17,fitted_magnitud:17,fitted_model:2,fitted_mu:17,fitted_mu_with_ref:17,fitted_param:[6,17],fitted_paramet:17,fitted_params_with_ref:17,fitted_pwv_effect:17,fitting_cli:18,fitting_pool:2,fittingpipelin:2,fix:[3,4,18],fixedrestransmiss:4,flag:20,flatlambdacdm:[3,5,6,8],flatsourc:15,float64:10,float64str1000float64str100:17,float64str2str12int32float32float32float32float32float32float32float32:16,flt:21,fluctuat:17,flux:[3,4,5,6,7,14,16,17],flux_with_pwv:16,flux_without_atm:14,flux_without_pwv:16,fluxcal:21,fluxcalerr:21,fluxerr:16,focu:15,follow:[4,9,15,16,17,20,21],foo:9,forc:[1,12,16,17],foremost:11,forget:20,fork:2,form:8,formal:21,format:[2,3,4,5,6,14,16,20],format_plasticc_sncosmo:[5,16],formatted_lc:[5,16],fortran:5,fortun:15,frac:17,fraction:12,frame:[4,6],framealpha:[12,14,15,16],frequent:14,from:[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21],from_suominet_receiv:[4,15,16],full:15,fulli:15,fundament:[13,17],further:2,futur:[15,20],gain:[3,5,16,17],galact:16,gener:[6,16,17,19],get:[4,5,7,8,15,16,17],get_available_cad:[5,16],get_available_typ:7,get_bandpass:[1,12,14],get_config_pwv_v:[8,17],get_data_dir:5,get_model_head:[5,16],get_ref_star_datafram:[7,12],get_stellar_spectra:[7,12],git:20,github:[11,19,20],given:[1,2,3,4,5,6,7,8,10,12,13,14,15,18,21],glass:1,global:19,goe:12,goettingen:19,good:15,gps_pwv:[15,16],gpsreceiv:[4,15,16],gracefulli:2,grei:[11,12,15],grid:16,grizi:17,gunzip:20,hand:[19,21],handl:[2,11,15],hardwar:[1,12],has:[5,9,11,17],hashabl:9,have:[3,6,15,16,17,20],head:[5,21],headach:20,header:[5,21],header_data:16,header_path:[5,16],height:4,help:1,here:[2,3,11,16,17,19],high:19,highest:4,highlight:6,hist:16,histogram:16,home:[8,20],horizon:16,host:[16,19,20],how:[11,12,13,15,17],howev:[4,11,17],http:[19,20],hubbl:[0,6],humid:[18,19],hundr:20,ibc:21,idea:15,ident:19,identifi:21,iloc:10,ilot:21,impact:13,implement:9,impos:[2,9],inch:6,includ:[3,4,6,7,8,11,13,16,17,19,20],incorpor:[10,16],increas:11,indefinit:20,independ:[4,17],index:[4,6,7,12,15,17,21],indic:2,individu:20,induc:14,inf:[2,18],inform:[10,11,16,21],initi:13,input:[5,10,15],insert:[12,14,15,16,17],instanc:[2,4,15,18],instanti:[2,4,9,20],instead:[4,15,16,17],intend:[10,21],interchang:3,interest:13,intern:[15,19],interp_norm_flux:7,interplai:17,interpol:[4,10,15],interpolated_pwv:15,interv:19,intrins:[0,6],involv:11,ipython:15,is_positive_airmass:16,issu:11,item:16,iter:[3,5,17],iter_lc_for_cadence_model:5,iter_lc_for_head:[5,16],iter_lcs_fixed_snr:[3,17],iter_light_curve_iter_with_ref:17,iter_lim:[2,18],januari:10,job:15,jupyt:13,keep:15,kei:6,kelvin:14,kept:11,kill:2,killsign:2,kind:15,know:20,kraken_2026:16,kraken_2044:16,kwarg:[8,15],label:[6,12,14,15,16],labelpad:[12,16],lambda:15,larger:6,largest:[12,17],lat:4,later:[14,17,19],latest:10,latitud:[0,4],launch:11,lc_data_set:16,lc_iter:5,lc_list:16,lc_simul:17,lc_tabl:7,least:15,leav:8,left:[12,14],left_ax:[12,14],left_twin_i:14,legaci:1,legend:[12,14,15,16],len:[1,8,16],length:[16,17],lens:[1,12],less:8,level:10,librari:[12,19],light:[2,3,4,5,7,8,13,17,21],light_curv:[3,5,8,17],light_curves_for_snr_plot:16,light_curves_with_ref:17,lightgrei:16,like:1,limit:[2,9,10,14,16],line:[4,6,14,16,20],linear:[10,14],linearli:[10,15],linestyl:[14,16],linewidth:[12,15],list:[2,4,5,6,7,8,14,16,17,19,20],llst:12,load:[5,7,16],loc:[12,14,15],local:5,locat:[4,16],log:14,lon:4,longitud:[0,4],look:[16,17],los:14,los_tick:14,los_xlim:14,lower:[8,12,14,18],lru_cach:9,lsst:[1,13,14,15,16,18],lsst_:1,lsst_atm:12,lsst_atm_norm:12,lsst_atmos_10:1,lsst_atmos_std:[1,12],lsst_detector:[1,12],lsst_filter:12,lsst_filter_:[1,12],lsst_hardware_:[1,12,14,16,17],lsst_hardware_r:17,lsst_len:1,lsst_lens:[1,12],lsst_m:1,lsst_mirror:[1,12],lsst_total_:[1,12],lsst_total_u:1,lsst_u_band:1,lsstdesc:20,m_b:8,m_nu:[3,5,6,8],made:[13,17],mag:[6,8,14,17],mag_dict:6,magnitud:[0,3,6,7,8,13,14,21],magsi:3,mai:[11,16],maintain:15,make:[5,12,15,17],manag:[2,20],mani:16,manipul:10,manual:[14,20],map:6,mask:2,master:20,match:[2,5],mater:0,matplotlib:[6,12,14,15,16,17],matter:[6,15],max:[14,16],max_queu:2,max_siz:9,maximum:[2,9],maxwav:14,md0:16,mdwarf:21,mean:[10,13,17],measur:[6,10,15,17,18],meet:11,memoiz:9,memori:[5,9,11],memorycach:9,meta:[2,5,16,17,21],metadata_conflict:16,meteorolog:19,meter:[0,4],method:[4,9,10],microsecond:10,milki:21,millibar:18,millimet:18,mimick:16,min:[14,16],minim:[11,12,17],minimum:11,minut:19,minwav:14,mira:21,mirror:[1,12],miss:[4,6,10,15],mix:20,mjd:[4,5,15,16,21],mjd_val:15,mjdfltfieldphotflagphotprobfluxcalfluxcalerrpsf_sig1sky_sigzeroptsim_magob:16,mkdir:[12,14,17,20],mnt:16,model:[2,3,5,6,8,12,13,14,16,17,19],model_for_sim:16,model_with_pwv:[16,17],model_without_pwv:[16,17],modeled_tran:12,modifi:7,modul:[0,11,17],moduli:[6,8],modulo:10,modulu:[6,8],moment:16,more:[4,10,11,13,14,16,17],motiv:17,mpc:[0,3,5,6,8],mu_pwv_effect:17,mu_pwv_effects_with_ref:17,multi:6,multipl:[2,6,16,17,18,19,21],must:[6,9],mwebv:21,name:[1,2,4,5,6,8,9,13,20,21],nan:[10,15],ndarrai:[4,6,7,8,10],nditer:6,nearbi:15,necessari:11,need:[1,15,16,19,20],neff:[3,5,6,8],negative_coord:16,nest:20,newli:1,next:[5,16,17],nois:[3,5],non:[12,16,17,21],none:[1,2,3,4,5,6,8,9,12,15,16],norm:14,norm_pwv:14,normal:[3,5,7,8,12,16,17,21],normalized_tran:12,notat:6,note:[12,17],notebook:[12,14,15,16,17],nuisanc:[0,6],num:6,num_lc:5,number:[2,3,5,6,10,11,16,18,21],numer:[4,17,18],numerical_error_offset:17,numerical_error_offset_ref:17,numpi:[9,10,12,14,15,16,17],numpy_arg:9,numpy_cach:9,numpyvar:4,ob0:[3,5,6,8],object:[0,2,3,4,5,9,10,11,12,14,15,17,21],obs:[3,4,8,16,17],observ:[2,3,4,5,6,7,11,13,16,17,18,21],observatori:[0,4,15,19],offici:21,oldest:9,om0:[3,5,6,8],omega:6,onc:[1,2,20],one:[8,16],ones:12,ones_lik:12,onli:[1,16,18,19,21],onlin:13,open:[11,16],optic:1,optim:12,option:[2,3,4,5,6,8,9,16,18],orang:11,order:[6,9,10],org:20,origin:[16,19],other:[3,8,10,19],our:[12,15,16,17],out:12,out_path:[2,18],outlin:[3,11,13,15,18,20,21],output:[2,11,18,20],outputdatamodel:2,over:[1,3,5,6,7,12,15,16,17,19],overal:[9,17],overlaid:6,overplot:16,overwrit:[1,5,16],own:[12,16],packag:[0,1,4,5,10,11,13,16,20],page:[20,21],pair:[5,21],panda:[7,10],pandas_obj:10,panel:6,paper:14,parallel:[2,11],param:[3,8,16,17],param_nam:16,paramet:[0,1,2,3,4,5,6,7,8,9,10,12,16,17,18],params_dict:6,parent:[0,12,14,17,19,21],pars:15,part:20,particular:[10,11],particularli:18,pass:[4,10,17],path:[2,5,8,12,14,15,16,17,18,19,20],pathlib:[12,14,17],pattern:4,pdf:[12,17],peak:[3,16],peakmjd:16,per:[4,12,15,18],percentag:18,perform:[11,19],period:[10,15],periodic_interpol:[10,15],phase:[3,6,16],phenomena:4,phot:[5,7,21],photflag:[5,21],photometr:21,photometri:[5,17,21],photprob:21,physic:[0,4,15,17],pick:15,pip:20,pipelin:[2,6,18],pisn:21,plastic_it:16,plasticc:20,plasticc_lc:[5,16],pleas:[1,11,20],plot:[12,14,15,16,17],plot_airmass_valid:16,plot_cosmology_fit:6,plot_delta_bb_mag:14,plot_delta_color:[6,17],plot_delta_mag_vs_pwv:6,plot_delta_mag_vs_z:6,plot_delta_mu:[6,17],plot_delta_x0:6,plot_derivative_mag_vs_z:6,plot_fitted_param:[6,17],plot_interpolation_valid:15,plot_lc:16,plot_lsst_atm:12,plot_lsst_filt:12,plot_magnitud:[6,17],plot_pwv_mag_effect:[6,17],plot_residuals_on_ski:6,plot_scale_factor:12,plot_snr_distribut:16,plot_spectral_templ:[6,17],plot_stellar_spectrum:12,plot_variable_pwv_sn_model:16,plot_year_pwv_vs_tim:[6,15],plt:[12,14,15,16,17],png:14,point:[3,5,16,18,21],polici:11,poly1d:12,polyfit:12,polynomi:12,pool:11,pool_siz:2,posit:19,positive_airmass:16,positive_coord:16,posixpath:8,possibl:11,postfix:21,practic:17,pre:[13,16],prebuilt:3,precipit:[4,13,14,15,19],precis:6,predefin:9,presenc:[8,17],pressur:[18,19],previou:17,primari:10,primarili:19,print:[1,2,5,9,10,12,15,16,17],prioriti:10,process:[2,11,15,17,18],processmanag:2,profil:[1,2,18,19],progress:[3,5,8],project:[11,13,15,16,19,20,21],propag:[4,16],propagationeffect:4,properti:[2,4],propog:[4,15],provid:[2,3,4,5,6,8,9,10,11,12,13,18,19,21],psf_sig1:21,ptrobs_max:21,ptrobs_min:21,publish:19,pull:15,pwv:[2,3,4,6,7,8,12,13,16,20],pwv_arr:[3,6,8,17],pwv_data:15,pwv_eff:14,pwv_eff_label:14,pwv_effect:16,pwv_interpol:4,pwv_kpno:[4,12,14,15,16,19],pwv_label:14,pwv_lo:[4,14],pwv_los_label:14,pwv_mag:14,pwv_model:[2,4,15,16],pwv_model_year:18,pwv_seri:[4,6],pwv_val:[14,17],pwv_zenith:[4,15],pwvmodel:[4,15,16],pyplot:[12,14,15,16,17],python:[5,9],quality_callback:2,question:[11,13],quick:[15,21],quickli:16,radian:16,rais:[2,11,15],random:[3,5],rang:[3,6,11,12,15,16],rate:18,ratio:3,rcparam:16,reader:11,real:[13,15],realist:[13,16],realiz:3,realize_lc:5,reason:[12,15],recalibr:7,receiv:4,receiver_id:18,recogn:4,record:[13,21],red:11,redder:17,reddest:12,redshift:[3,6,8,17],reduc:4,ref:17,ref_df:12,ref_pwv:[6,8],ref_star:[2,18],refer:[2,6,7,8,11,14,18,19,21],reference_idx:17,reference_pwv:17,reference_pwv_config:17,reference_star:[12,17],reflect:[2,3],reformat:16,region:12,regist:[1,10],register_decam_filt:1,register_lsst_filt:[1,12,14,16,17],register_sncosmo_filt:1,rel:[6,7,8,12,14,18,19],relat:[16,19],reli:12,remov:9,remove_column:17,render:13,replac:19,report:12,repositori:[19,20],repres:[0,6,11,12,15,19,21],represent:[6,15],reproduc:6,requir:[11,19,20],res:[12,14],resampl:[10,15],resample_data_across_year:[10,15],resampled_pwv:15,rescal:[5,14],research:[11,19],resembl:4,reset_index:15,residu:6,resolut:[4,6,12,14,15,19],respect:17,respons:[1,8,12,19],restart:20,result:[1,2,4,6,11],retri:20,retriev:[1,5,7,8],review:11,right:[4,6,12,15],right_ax:[12,14],right_twin_i:14,rizi:14,rom:15,rotat:[12,16],rough:12,routin:9,row:[2,6,21],rrlyra:21,rubin:[0,4,15],run:[2,5,11,16,18,20],run_async:2,runner:8,runtim:1,salt2:[3,4,6,16,17,18],same:[4,8,10,16,17],samp_wav:4,sampl:[14,16,17,19],saniti:17,satur:14,save:21,savefig:[12,14,17],scalar:4,scale:[12,14,15,16],scat:16,scatter:[3,5,15,16],sci_not:6,scienc:[11,19],scientif:[6,11,13],scipi:12,scratch:[16,20],script:1,sdss:3,sdssg:3,sdssi:3,sdssr:3,sdssu:3,season:[4,15],seasonal_averag:4,seasonal_prop:15,seasonalpwvtran:[4,15],sec:[4,6],second:10,section:11,sed:[12,14,17],sed_with_pwv:14,see:[10,11,17],select:[13,15],self:9,sensit:[1,17],separ:12,seri:[4,6,7,10,15],server:16,servic:19,set:[2,3,4,6,13,14,15,17,19,20],set_label:16,set_titl:[12,16],set_xlabel:[12,14,16],set_xlim:[12,14,16],set_xtick:14,set_xticklabel:14,set_ylabel:[12,14,16],set_ylim:[12,14],sever:[13,20],shade:11,shape:[8,17],sharei:[12,14,16],sharex:[12,16],should:[2,5,6,9,17,21],show:[3,6,8,12,14,15,16],shown:[11,15,20],sight:[4,14,16],signal:[2,3],signific:15,silent:16,sim:16,sim_magob:21,sim_model:[2,8],sim_model_nam:21,sim_peakmjd:16,sim_pool_s:18,sim_redshift_cmb:16,sim_salt2c:16,sim_salt2x0:16,sim_salt2x1:16,sim_sourc:18,sim_vari:18,similar:[2,4,10,16,17],similarli:16,simpl:15,simplic:17,simul:[2,3,5,6,7,8,11,13,14,17,20,21],simulate_lc:3,simulate_lc_fixed_snr:3,simulation_pool:2,sinc:[5,16,17],singl:[2,3,6,10,15],size:[6,9,12,17],skip:2,sky:21,skycoord:16,skynois:[3,5,16,17],slope:[6,8,17],slope_arr:6,slope_end:17,slope_end_idx:17,slope_start:17,slope_start_idx:17,slow:5,slsn:21,smooth:15,smoother:14,sn_coord:16,sn_magnitud:17,sn_model:3,sn_model_fit:2,sn_model_sim:2,snana:[5,16,21],snat_sim:[11,12,13,14,15,16,17,20],sncc:21,sncosmo:[1,2,3,4,5,6,8,12,14,15,16,17,18],sncosmo_lc:16,sne:[0,3,4,5,6,16,17,18],snia:21,snid:[16,21],snmodel:[2,3,4,5,8,16],snr:[3,16],softwar:[11,20],solstic:4,solut:9,some:[15,17],sometim:15,sort_index:15,sourc:[1,2,3,4,5,6,7,8,9,10,11,13,16,19],space:1,span:15,spawn:18,specif:[12,13],specifi:[3,5,9,10,18,20],spectra:12,spectral:[3,4,6,7,12,16,18,19],spectral_templ:17,spectral_typ:[7,17],spectrum:[7,12,14],spectyp:12,spring:15,srd:[1,19],stage:11,standard:[3,12],star:[2,7,18],start:[2,8,12,16,17,21],startup:20,stat:12,state:6,static_prop:15,staticpwvtran:[4,15,17],statist:[3,15],std:16,stdev:16,stellar_scale_factor:12,stellar_spectra:19,stellar_spectrum:12,step:[11,15,19,20],still:15,storag:20,store:[2,5,11],str:[1,2,3,4,5,6,7,8,12,16],stretch:[6,8,17],string:[2,6,21],subject:11,subplot:[6,12,14,16],subset:[10,20],subtract:7,subtyp:21,suffer:4,suit:20,summar:[0,5],summari:[19,21],summed_residu:12,summer:15,suominet:[4,13,15,18,19],supernova:[2,3,4,6,8,11,13,16,18,19,21],supernova_model:4,supp_year:[4,10],supplement:[4,10,15],supplementari:10,supplemented_data:[10,15],supplemented_pwv:15,support:[3,4,9,10,11],suptitl:17,sure:17,surfac:18,survei:[1,11,15,19],svo:19,synchron:2,synthet:17,sys:[12,14,15,16,17],system:3,tabl:[0,2,3,5,7,8,16,17,21],tabul:[6,8,16,17],tabular:2,tabulate_fiducial_mag:[8,17],tabulate_mag:[8,17],tabulated_delta_mag:17,tabulated_fiducial_mag:17,tabulated_mag:17,tabulated_magnitud:17,tabulated_pwv_effect:17,tabulated_slop:17,take:[12,16,17,19,20],taken:[6,10,13,15,19,21],tar:20,target:[3,4,5,21],task:[10,11],tcmb0:[3,5,6,8],tde:21,team:19,technic:11,techniqu:11,telolo:[15,19],temp:14,temperatur:[14,18,19],templat:[3,4,6,16,18],tempor:[15,16,17],termin:2,test:[15,20],test_model:15,text:17,than:[6,8],thei:[1,10,15],them:20,theoret:13,thi:[2,4,5,8,10,11,12,13,14,15,16,17,18,19,20,21],thing:16,think:[15,17],those:13,three:16,through:[4,10,15,17],throughput:[1,6,19],thu:5,tight:17,tight_layout:[14,16],time:[1,3,4,6,10,11,13,15,16,19],time_format:[4,15],time_seri:15,timebandzpzpsi:17,timedelta:15,timeout:20,timezon:15,titl:15,tol:12,too:15,total:[1,12,18],touch:20,tran:[1,12],transit:15,transmiss:[1,4,6,12,14,16,17],transmission_r:4,trapz:14,treat:9,trend:17,tri:20,trigger:21,tsu:[10,15],tsuaccessor:10,tupl:[2,6,8,12],twin_ax:12,twini:14,twinx:12,two:15,txt:20,type:[1,2,3,4,5,6,7,8,9,10,11,12,13,16,19,21],ugrizi:[1,12,16],ulen:21,ultim:11,uncertainti:3,unchang:8,uncorrect:17,under:10,underli:[4,15],understand:[11,17],unfortun:15,uniform:[3,17],uniformli:15,union:[2,3,4,5,6,7,8,10],uniqu:21,unit:[0,3,4,6,14,16,18],unlik:[4,17],unset:20,until:2,updat:16,upper:[8,12,14,15,18],url:20,usag:[13,18],use:[2,3,4,5,6,7,8,14,15,16,17,18,21],used:[0,2,4,5,6,8,10,14,15,16,18,19,20,21],useful:18,user:15,using:[2,3,4,5,7,10,13,15,16,19,20,21],utc:15,util:[7,15,20],v1_transmiss:[4,12,14],val:8,valid:[15,16,19],valu:[0,1,2,3,4,5,6,7,8,10,14,15,16,17,18,19,21],vapor:[4,13,14,15,19],vari:[2,4,8,15,18],variabl:[4,12,13,15,16,18,19,20],variable_prop:15,variablepropagationeffect:4,variablepwvtran:[4,15,16],variat:[4,6,17],varieti:19,variou:[4,19,20],vera:0,verbos:[3,5,8,16,20],verifi:16,version:[13,17],via:[5,8,20],visual:[6,11,16,17],vmax:16,vmin:16,vparam:[2,8,17,18],vro:4,vro_altitud:0,vro_latitud:0,vro_longitud:0,vstack:16,wai:21,wait:2,wait_for_exit:2,want:20,water:[4,13,14,15,19],wave:[1,4,12,14,16],wave_arr:6,wave_rang:12,wavelength:[1,4,6,7,12,14,16],weather:[10,13,15],weather_data:15,weather_for_year:15,websit:19,welcom:11,well:11,were:[5,12,13,19,21],wget:20,what:[5,10,16,17],when:[1,2,3,4,5,8,9,10,16,17,18],where:[5,20,21],wherev:11,whether:1,which:[2,3,4,15,16,18],window:6,winter:15,within:[1,10,11,18],without:[1,2,3,4,16],wont:17,work:[5,8,11,17,20],worker:18,world:[13,15],would:10,wrap_at:16,wrapper:9,wrestl:14,write:2,written:[11,13,16],x_1:8,x_arr:9,x_val:15,xlabel:15,xlim:15,xvzf:20,y_arr:9,yaml:8,year:[1,4,6,10,15,16],yield:[3,5,17],ylabel:15,ylim:[14,15],ymin:14,you:[5,20],z_arr:[3,6,8,17],z_val:17,zenith:[4,15,16,18,19],zenodo:[19,20],zero:[3,4,5,16,17,21],zeropt:21,zip:[16,17],zorder:12,zpsy:3},titles:["snat_sim.constants","snat_sim.filters","snat_sim.fitting_pipeline","snat_sim.lc_simulation","snat_sim.models","snat_sim.plasticc","snat_sim.plotting","snat_sim.reference_stars","snat_sim.sn_magnitudes","snat_sim.utils.caching","snat_sim.utils.time_series","Impact of Chromatic Effects on LSST SNe Ia","LSST Filters","Notebook Summaries","Effective PWV on a Black Body","PWV Modeling","Simulating Light-Curves for a Given Cadence","PWV effects on SN Magnitude","Command Line Interface","Data Provenance","Installation and Setup","PLaSTICC Data Model"],titleterms:{"function":6,Adding:16,The:[12,15,16],With:17,api:9,appar:17,approach:15,argument:18,atm:12,atmospher:[16,19],autom:15,baselin:12,black:14,bodi:14,build:15,cach:9,cadenc:16,chang:17,chromat:11,color:17,command:18,configur:20,constant:0,contribut:11,ctio:19,curv:[16,18,19,20],data:[16,19,21],distanc:17,doc:[1,2,3,4,5,6,7,8,10],download:20,effect:[11,14,15,16,17],environ:20,exampl:[1,2,3,4,5,9,10],fiduci:12,file:21,filter:[1,12,19],fit:[17,18],fitting_pipelin:2,flux:12,format:21,given:16,impact:[11,17],instal:20,interfac:18,lc_simul:3,light:[16,18,19,20],line:18,lsst:[11,12,19],magnitud:17,measur:19,model:[4,15,18,21],modul:[1,2,3,4,5,6,7,8,9,10],modulu:17,name:18,notebook:13,organiz:21,overview:11,pipelin:11,plasticc:[5,16,21],plot:6,propag:15,proven:19,pwv:[14,15,17,18],refer:17,reference_star:7,rel:17,result:17,set:12,setup:20,sim:20,simul:[16,18,19],sn_magnitud:8,snat_sim:[0,1,2,3,4,5,6,7,8,9,10],sne:11,sourc:20,spectra:19,spectral:17,star:17,stellar:[12,19],summari:[4,6,13],templat:17,time_seri:10,usag:[1,2,3,4,5,9,10,11],using:17,util:[9,10],without:17,your:20}})
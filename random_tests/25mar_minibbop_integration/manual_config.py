sessions1 = [
    # {
    #     "session_dir": "/data/big_rim/rsync_dcc_sum/Oct3V1mini_sorted/202401015-PMC-r2/customEntValHere/2024_11_07/13_04_45/My_V4_Miniscope",
    #     "max_wnd": 15,
    #     "diff_thres": 3.0,
    # },

    {
        "session_dir": "/data/big_rim/rsync_dcc_sum/Oct3V1mini_sorted/202401015-PMC-r2/customEntValHere/2024_11_07/13_04_45/My_V4_Miniscope",
        "script": "cleanup_session_keep_multi.py",
       "keep_combination": "wnd1500_stp700_max15_diff5.0_pnrauto"
    #    "wnd1500_stp700_max15_diff3.5_pnrauto,wnd1500_stp700_max15_diff5.0_pnrauto,wnd1500_stp700_max15_diff4.0_pnrauto," #"wnd1500_stp700_max15_diff3.5_pnrauto"
    },
        {
        "session_dir": "/data/big_rim/rsync_dcc_sum/Oct3V1mini_sorted/20240819-V1-r1/customEntValHere/2024_10_17/14_09_19/My_V4_Miniscope",
        "script": "cleanup_session_keep_multi.py",
       "keep_combination": "wnd1500_stp700_max25_diff3.5_pnr1.1" #"wnd1500_stp700_max15_diff3.5_pnrauto"
    },


    {
        "session_dir": "/data/big_rim/rsync_dcc_sum/Oct3V1mini_sorted/20240819-V1-r1/customEntValHere/2024_10_31/11_31_56/My_V4_Miniscope",
     "script": "cleanup_session.py",
       "keep_combination": "wnd1500_stp700_max25_diff5.0_pnr1.1" #"wnd1500_stp700_max15_diff3.5_pnrauto"

        

    },

    

    {
        "session_dir": "/data/big_rim/rsync_dcc_sum/Oct3V1mini_sorted/20240819-V1-r1/customEntValHere/2024_10_31/11_50_32/My_V4_Miniscope",
             "script": "cleanup_session.py",
       "keep_combination": "wnd1500_stp700_max25_diff3.5_pnr1.1" #"wnd1500_stp700_max15_diff3.5_pnrauto"

        

    },
       {
        "session_dir": "/data/big_rim/rsync_dcc_sum/Oct3V1mini_sorted/20240819-V1-r1/customEntValHere/2024_11_01/14_40_33/My_V4_Miniscope",
             "script": "cleanup_session.py",
       "keep_combination": "wnd1500_stp700_max25_diff5.0_pnr1.1" 

    
    },

          {
        "session_dir": "/data/big_rim/rsync_dcc_sum/Oct3V1mini_sorted/20240819-V1-r1/customEntValHere/2024_11_01/14_57_45/My_V4_Miniscope",
        "script": "cleanup_session.py", 
       "keep_combination": "wnd1500_stp700_max25_diff5.0_pnrauto"
        },

              {
        "session_dir": "/data/big_rim/rsync_dcc_sum/Oct3V1mini_sorted/202409-V1-r1/customEntValHere/2024_11_01/14_57_45/My_V4_Miniscope",
        "script": "cleanup_session.py", 
       "keep_combination": "wnd1500_stp700_max25_diff5.0_pnrauto"
        },


    # {
    #     "session_dir": "",
    #     "max_wnd": 15,
    #     "diff_thres": 3.0,
    # },
    #     {
    #     "session_dir": "",
    #     "script": "param_search_driver_simple.py"
    # },
      # {
    #     "session_dir": "",
    #     "script": "cleanup_session.py", #"cleanup_session_keep_multi.py",
    #    "keep_combination": "" #"wnd1500_stp700_max15_diff3.5_pnrauto"
    #     },
]

sessions2 = [
    {
        "session_dir": "/data/big_rim/rsync_dcc_sum/Oct3V1mini_sorted/202401015-PMC-r2/customEntValHere/2024_11_07/13_04_45/My_V4_Miniscope",
        "max_wnd": 15,
        "diff_thres": 3.0,
    },

    {
        "session_dir": "/data/big_rim/rsync_dcc_sum/Oct3V1mini_sorted/20240819-V1-r1/customEntValHere/2024_10_17/14_09_19/My_V4_Miniscope",
        "max_wnd": 25,
        "diff_thres": 3.0,
    },

        {
        "session_dir": "/data/big_rim/rsync_dcc_sum/Oct3V1mini_sorted/20240819-V1-r1/customEntValHere/2024_10_17/14_09_19/My_V4_Miniscope",
        "max_wnd": 25,
        "diff_thres": 5.0,
        "param_pnr_refine.thres": "auto",
        

    },

    

    {
        "session_dir": "/data/big_rim/rsync_dcc_sum/Oct3V1mini_sorted/20240819-V1-r1/customEntValHere/2024_10_31/11_50_32/My_V4_Miniscope",
        "max_wnd": 25,
        "diff_thres": 3.0,
        "param_pnr_refine.thres": "auto",
        

    },
       {
        "session_dir": "/data/big_rim/rsync_dcc_sum/Oct3V1mini_sorted/20240819-V1-r1/customEntValHere/2024_11_01/14_40_33/My_V4_Miniscope",
        "max_wnd": 25,
        "diff_thres": 3.0,
        "param_pnr_refine.thres": "auto",
        

    },
#

   
    # {
    #     "session_dir": "/data/big_rim/rsync_dcc_sum/Oct3V1mini_sorted/202401015-PMC-r2/customEntValHere/2024_11_07/13_04_45/My_V4_Miniscope",
    #     "script": "cleanup_session_keep_multi.py",
    #    "keep_combination": "wnd1500_stp700_max15_diff3.5_pnrauto,wnd1500_stp700_max15_diff5.0_pnrauto,wnd1500_stp700_max15_diff4.0_pnrauto," #"wnd1500_stp700_max15_diff3.5_pnrauto"
    # },
    # {
    #     "session_dir": "",
    #     "max_wnd": 15,
    #     "diff_thres": 3.0,
    # },
    #     {
    #     "session_dir": "",
    #     "script": "param_search_driver_simple.py"
    # },
      # {
    #     "session_dir": "",
    #     "script": "cleanup_session.py", #"cleanup_session_keep_multi.py",
    #    "keep_combination": "" #"wnd1500_stp700_max15_diff3.5_pnrauto"
    #     },
]

future_sessions = [

    {
        "session_dir": "/data/big_rim/rsync_dcc_sum/Oct3V1mini_sorted/20241224PMCLE1/customEntValHere/2025_02_13/11_07_37/My_V4_Miniscope",
        "max_wnd": 15,
        "diff_thres": 3.0,
    },

    {
        "session_dir": "/data/big_rim/rsync_dcc_sum/Oct3V1mini_sorted/202401015-PMC-r2/customEntValHere/2024_11_07/13_04_45/My_V4_Miniscope",
        "script": "param_search_driver_simple.py"
    },

    {
        "session_dir": "/data/big_rim/rsync_dcc_sum/Oct3V1mini_sorted/202401015-PMC-r2/customEntValHere/2024_11_07/13_25_34/My_V4_Miniscope",
        "script": "param_search_driver_simple.py"
    },

    
    #we don't need clean up unless they need to be optimized becuase there is corrsponding recordings i guess. stupid.
    # {
    #    "session_dir": "/data/big_rim/rsync_dcc_sum/Oct3V1mini_sorted/20240819-V1-r1/customEntValHere/2024_10_08/19_44_49/My_V4_Miniscope",
    #     "script": "cleanup_session.py",
    #    "keep_combination": "wnd7000_stp700_max25_diff3.5_pnrauto" #"wnd1500_stp700_max15_diff3.5_pnrauto"
 
    # },
    # {
    #     "session_dir": "/data/big_rim/rsync_dcc_sum/Oct3V1mini_sorted/20240819-V1-r1/customEntValHere/2024_10_08/20_00_58/My_V4_Miniscope",
    #     "script": "cleanup_session.py",
    #    "keep_combination": "wnd7000_stp700_max25_diff3.5_pnr1.1" #"wnd1500_stp700_max15_diff3.5_pnrauto"


    # },
    #     {
    #     "session_dir": "rsync_dcc_sum/Oct3V1mini_sorted/20240819-V1-r1/customEntValHere/2024_10_08/20_32_38/My_V4_Miniscope",
    #     "script": "cleanup_session.py",
    #    "keep_combination": "" #"wnd1500_stp700_max15_diff3.5_pnrauto"
    #     },
    # {
    #     "session_dir": "/data/big_rim/rsync_dcc_sum/Oct3V1mini_sorted/20240819-V1-r1/customEntValHere/2024_10_17/14_25_32/My_V4_Miniscope",
    #     "script": "cleanup_session.py",
    #     "keep_combination": "" 
    # },
    {
        
    },
    # {
    #     "session_dir": "",
    #     "script": "cleanup_session.py",
    #    "keep_combination": "" #"wnd1500_stp700_max15_diff3.5_pnrauto"
    #     },

      {
        "session_dir": "/data/big_rim/rsync_dcc_sum/Oct3V1mini_sorted/20241224PMCLE1/customEntValHere/2025_02_13/11_07_37/My_V4_Miniscope",
        "max_wnd": 15,
        "diff_thres": 3.0,
    },

    {
        "session_dir": "/data/big_rim/rsync_dcc_sum/Oct3V1mini_sorted/20240916-V1-R1/customEntValHere/2024_10_14/16_54_16/My_V4_Miniscope",
        "script": "param_search_driver.py"
    },
    {
        "session_dir": "/data/big_rim/rsync_dcc_sum/Oct3V1mini_sorted/20240916-V1-R2/customEntValHere/2024_10_25/14_31_19/My_V4_Miniscope",
        "script": "param_search_driver.py"
    },
    {
        "session_dir": "/data/big_rim/rsync_dcc_sum/Oct3V1mini_sorted/20241002-pmc-r2/customEntValHere/2024_10_25/16_27_28/My_V4_Miniscope",
        "script": "param_search_driver.py"
    },
    {
        "session_dir": "/data/big_rim/rsync_dcc_sum/Oct3V1mini_sorted/20241002-PMC-R1/customEntValHere/2024_10_25/15_02_11/My_V4_Miniscope",
        "script": "param_search_driver.py"
    },
    {
        "session_dir": "/data/big_rim/rsync_dcc_sum/Oct3V1mini_sorted/20241002-PMC-R1/customEntValHere/2024_10_24/13_07_09/My_V4_Miniscope",
        "script": "param_search_driver.py"
    },
    {
        "session_dir": "/data/big_rim/rsync_dcc_sum/Oct3V1mini_sorted/20241001-PMC-r2/customEntValHere/2024_10_24/16_20_26/My_V4_Miniscope",
        "script": "param_search_driver.py"
    },
    {
        "session_dir": "/data/big_rim/rsync_dcc_sum/Oct3V1mini_sorted/20241001-PMC-r2/customEntValHere/2024_10_24/15_27_55/My_V4_Miniscope",
        "script": "param_search_driver.py"
    },
    {
        "session_dir": "/data/big_rim/rsync_dcc_sum/Oct3V1mini_sorted/202401015-PMC-r2/customEntValHere/2024_11_06/16_57_35/My_V4_Miniscope",
        "script": "param_search_driver.py"
    },
    {
        "session_dir": "/data/big_rim/rsync_dcc_sum/Oct3V1mini_sorted/202401015-PMC-r2/customEntValHere/2024_11_07/13_04_45/My_V4_Miniscope",
        "script": "param_search_driver.py"
    },
    {
        "session_dir": "/data/big_rim/rsync_dcc_sum/Oct3V1mini_sorted/202401015-PMC-r2/customEntValHere/2024_11_07/13_25_34/My_V4_Miniscope",
        "script": "param_search_driver.py"
    },
    {
        "session_dir": "/data/big_rim/rsync_dcc_sum/Oct3V1mini_sorted/20240819-V1-r1/customEntValHere/2024_10_17/14_09_19/My_V4_Miniscope",
        "script": "param_search_driver.py"
    },
    {
        "session_dir": "/data/big_rim/rsync_dcc_sum/Oct3V1mini_sorted/20240819-V1-r1/customEntValHere/2024_10_31/11_31_56/My_V4_Miniscope",
        "script": "param_search_driver.py"
    },
    {
        "session_dir": "/data/big_rim/rsync_dcc_sum/Oct3V1mini_sorted/20240819-V1-r1/customEntValHere/2024_10_31/11_50_32/My_V4_Miniscope",
        "script": "param_search_driver.py"
    },
    {
        "session_dir": "/data/big_rim/rsync_dcc_sum/Oct3V1mini_sorted/20240819-V1-r1/customEntValHere/2024_11_01/14_40_33/My_V4_Miniscope",
        "script": "param_search_driver.py"
    },
    {
        "session_dir": "/data/big_rim/rsync_dcc_sum/Oct3V1mini_sorted/20240819-V1-r1/customEntValHere/2024_11_01/14_57_45/My_V4_Miniscope",
        "script": "param_search_driver.py"
    },
    {
        "session_dir": "/data/big_rim/rsync_dcc_sum/Oct3V1mini_sorted/20240910-V1-R/customEntValHere/2024_11_01/13_09_59/My_V4_Miniscope",
        "script": "param_search_driver.py"
    },
    {
        "session_dir": "/data/big_rim/rsync_dcc_sum/Oct3V1mini_sorted/20240916-V1-R1/customEntValHere/2024_10_14/14_54_12/My_V4_Miniscope",
        "script": "param_search_driver.py"
    },
    {
        "session_dir": "/data/big_rim/rsync_dcc_sum/Oct3V1mini_sorted/20240916-V1-R1/customEntValHere/2024_10_14/16_38_19/My_V4_Miniscope",
        "script": "param_search_driver.py"
    },
    {
        "session_dir": "/data/big_rim/rsync_dcc_sum/Oct3V1mini_sorted/20241002-pmc-r2/customEntValHere/2024_10_25/15_50_39/My_V4_Miniscope",
        "script": "param_search_driver.py"
    },
    {
        "session_dir": "/data/big_rim/rsync_dcc_sum/Oct3V1mini_sorted/20241002-pmc-r2/customEntValHere/2024_10_25/17_09_14/My_V4_Miniscope",
        "script": "param_search_driver.py"
    }
]
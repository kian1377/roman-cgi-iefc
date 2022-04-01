import poppy
from poppy.poppy_core import PlaneType

from pathlib import Path

from . import cgi
opddir = cgi.cgi_dir/'spc-spec-opds'

opdunits = 'meters'

primary_opd = poppy.FITSOpticalElement('Primary OPD', opd=str(opddir/'roman_phasec_PRIMARY_synthetic_phase_error_V1.0.fits'),
                                       opdunits=opdunits, planetype=PlaneType.intermediate)

secondary_opd = poppy.FITSOpticalElement('Secondary OPD',opd=str(opddir/'roman_phasec_SECONDARY_synthetic_phase_error_V1.0.fits'),
                                         opdunits=opdunits, planetype=PlaneType.intermediate)

poma_fold_opd = poppy.FITSOpticalElement('POMA-Fold OPD', opd=str(opddir/'roman_phasec_POMAFOLD_measured_phase_error_V1.1.fits'),
                                         opdunits=opdunits,planetype=PlaneType.intermediate)

m3_opd = poppy.FITSOpticalElement('M3 OPD', opd=str(opddir/'roman_phasec_M3_measured_phase_error_V1.1.fits'), 
                                  opdunits=opdunits, planetype=PlaneType.intermediate)

m4_opd = poppy.FITSOpticalElement('M4 OPD', opd=str(opddir/'roman_phasec_M4_measured_phase_error_V1.1.fits'), 
                                  opdunits=opdunits, planetype=PlaneType.intermediate)

m5_opd = poppy.FITSOpticalElement('M5 OPD', opd=str(opddir/'roman_phasec_M5_measured_phase_error_V1.1.fits'),
                                  opdunits=opdunits, planetype=PlaneType.intermediate)

tt_fold_opd = poppy.FITSOpticalElement('TT-Fold OPD', opd=str(opddir/'roman_phasec_TTFOLD_measured_phase_error_V1.1.fits'), 
                                       opdunits=opdunits, planetype=PlaneType.intermediate)

fsm_opd = poppy.FITSOpticalElement('FSM OPD', opd=str(opddir/'roman_phasec_LOWORDER_phase_error_V2.0.fits'),
                                   opdunits=opdunits, planetype=PlaneType.intermediate)

oap1_opd = poppy.FITSOpticalElement('OAP1 OPD', opd=str(opddir/'roman_phasec_OAP1_phase_error_V3.0.fits'),
                                    opdunits=opdunits, planetype=PlaneType.intermediate)

focm_opd = poppy.FITSOpticalElement('FOCM OPD', opd=str(opddir/'roman_phasec_FCM_EDU_measured_coated_phase_error_V2.0.fits'), 
                                    opdunits=opdunits, planetype=PlaneType.intermediate)

oap2_opd = poppy.FITSOpticalElement('OAP2 OPD', opd=str(opddir/'roman_phasec_OAP2_phase_error_V3.0.fits'), 
                                    opdunits=opdunits, planetype=PlaneType.intermediate)

dm1_opd = poppy.FITSOpticalElement('DM1 OPD', opd=str(opddir/'roman_phasec_DM1_phase_error_V1.0.fits'), 
                                   opdunits=opdunits, planetype=PlaneType.intermediate)

dm2_opd = poppy.FITSOpticalElement('DM2 OPD', opd=str(opddir/'roman_phasec_DM2_phase_error_V1.0.fits'),
                                   opdunits=opdunits, planetype=PlaneType.intermediate)

oap3_opd = poppy.FITSOpticalElement('OAP3 OPD', opd=str(opddir/'roman_phasec_OAP3_phase_error_V3.0.fits'), 
                                    opdunits=opdunits, planetype=PlaneType.intermediate)

fold3_opd = poppy.FITSOpticalElement('F3 OPD',opd=str(opddir/'roman_phasec_FOLD3_FLIGHT_measured_coated_phase_error_V2.0.fits'), 
                                     opdunits=opdunits, planetype=PlaneType.intermediate)

oap4_opd = poppy.FITSOpticalElement('OAP4 OPD', opd=str(opddir/'roman_phasec_OAP4_phase_error_V3.0.fits'), 
                                    opdunits=opdunits, planetype=PlaneType.intermediate)

pupil_mask_opd = poppy.FITSOpticalElement('SPM OPD', opd=str(opddir/'roman_phasec_PUPILMASK_phase_error_V1.0.fits'), 
                                          opdunits=opdunits, planetype=PlaneType.intermediate)

oap5_opd = poppy.FITSOpticalElement('OAP5 OPD', opd=str(opddir/'roman_phasec_OAP5_phase_error_V3.0.fits'), 
                                    opdunits=opdunits, planetype=PlaneType.intermediate)

oap6_opd = poppy.FITSOpticalElement('OAP6 OPD', opd=str(opddir/'roman_phasec_OAP6_phase_error_V3.0.fits'), 
                                    opdunits=opdunits, planetype=PlaneType.intermediate)

oap7_opd = poppy.FITSOpticalElement('OAP7 OPD', opd=str(opddir/'roman_phasec_OAP7_phase_error_V3.0.fits'),
                                    opdunits=opdunits, planetype=PlaneType.intermediate)

oap8_opd = poppy.FITSOpticalElement('OAP8 OPD', opd=str(opddir/'roman_phasec_OAP8_phase_error_V3.0.fits'), 
                                    opdunits=opdunits, planetype=PlaneType.intermediate)

filter_opd = poppy.FITSOpticalElement('Filter OPD', opd=str(opddir/'roman_phasec_FILTER_phase_error_V1.0.fits'), 
                                      opdunits=opdunits, planetype=PlaneType.intermediate)

lens_opd = poppy.FITSOpticalElement('LENS OPD', opd=str(opddir/'roman_phasec_LENS_phase_error_V1.0.fits'), 
                                    opdunits=opdunits, planetype=PlaneType.intermediate)







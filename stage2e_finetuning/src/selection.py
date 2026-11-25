"""Predeclared development-only clean-regression and parent-stability gates."""
def eligible(candidate,baseline,require_improvement=True):
 p=candidate['positive_development'];b=baseline['positive_development']
 c=candidate['clean_validation'];bc=baseline['clean_validation']
 long=candidate['long_positive'];blong=baseline['long_positive']
 historical_ok=True
 if 'clean_historical_regression' in candidate:
  hc=candidate['clean_historical_regression'];hb=baseline['clean_historical_regression']
  historical_ok=hc['cer']<=hb['cer']+.001 and hc['exact_accuracy']>=hb['exact_accuracy']-.01
 return historical_ok and (p['cer']<b['cer']-1e-12 if require_improvement else p['cer']<=b['cer']+1e-12) and c['cer']<=bc['cer']+.001 and c['exact_accuracy']>=bc['exact_accuracy']-.01 and p['macro_parent_cer']<=b['macro_parent_cer']+1e-12 and (long['cer'] is None or long['cer']<=blong['cer']+.005)
def rank(candidate):
 return (candidate['positive_development']['cer'],candidate['positive_development']['macro_parent_cer'],candidate['clean_validation']['cer'])


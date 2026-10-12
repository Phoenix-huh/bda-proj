"""AQI categories (CPCB scale) and plain-language health advice."""

BREAKPOINTS = [(50, 'Good'), (100, 'Satisfactory'), (200, 'Moderate'),
               (300, 'Poor'), (400, 'Very Poor')]

COLORS = {'Good': '#009865', 'Satisfactory': '#58b21d', 'Moderate': '#e6b800',
          'Poor': '#f28c28', 'Very Poor': '#e03a3a', 'Severe': '#7e0023'}

ADVICE = {
    'Good': dict(
        summary='Air quality is healthy.',
        general='Enjoy normal outdoor activities.',
        sensitive='No special precautions needed.',
        outdoor='All outdoor exercise is fine.',
        mask='Not needed.'),
    'Satisfactory': dict(
        summary='Acceptable air quality; a few very sensitive people may notice mild effects.',
        general='Normal activities are fine.',
        sensitive='People with asthma or heart conditions should watch for symptoms.',
        outdoor='Outdoor exercise is fine for most people.',
        mask='Not needed.'),
    'Moderate': dict(
        summary='Noticeable pollution; sensitive groups may have breathing discomfort.',
        general='Reduce long, intense outdoor exertion if you feel discomfort.',
        sensitive='Children, older adults and people with lung or heart disease should limit prolonged outdoor activity.',
        outdoor='Prefer lighter exercise; avoid heavy traffic corridors.',
        mask='Optional for sensitive people near traffic.'),
    'Poor': dict(
        summary='Prolonged exposure can cause discomfort for many people.',
        general='Limit prolonged outdoor exertion.',
        sensitive='Sensitive groups should stay indoors as much as possible and keep medication handy.',
        outdoor='Move workouts indoors or shorten them.',
        mask='A well-fitted N95/FFP2 mask is advisable outdoors.'),
    'Very Poor': dict(
        summary='Likely to affect breathing with prolonged exposure; worse for people with existing conditions.',
        general='Avoid outdoor exertion; keep windows closed.',
        sensitive='Stay indoors; consider an air purifier; contact a doctor if symptoms worsen.',
        outdoor='Avoid outdoor exercise.',
        mask='Wear an N95/FFP2 mask if you must go out.'),
    'Severe': dict(
        summary='Hazardous: even healthy people may be affected, even during light activity.',
        general='Stay indoors with windows closed; avoid all outdoor exertion.',
        sensitive='High risk: stay indoors, use a purifier, seek medical help for breathing difficulty or chest pain.',
        outdoor='No outdoor exercise.',
        mask='N95/FFP2 mask for any necessary trip outside.'),
}

CIGARETTE_NOTE = ('Rough rule of thumb (Berkeley Earth): about 22 ug/m3 of PM2.5 over a day '
                  'is comparable to one cigarette. It is an illustration, not a medical measurement.')


def aqi_bucket(aqi):
    for limit, name in BREAKPOINTS:
        if aqi <= limit:
            return name
    return 'Severe'


def advisory_for(aqi, pm25=None):
    bucket = aqi_bucket(aqi)
    adv = dict(ADVICE[bucket])
    adv.update(bucket=bucket, color=COLORS[bucket], aqi=round(float(aqi)))
    if pm25 is not None and pm25 >= 0:
        adv['cigarettes_per_day'] = round(float(pm25) / 22.0, 1)
        adv['cigarette_note'] = CIGARETTE_NOTE
    else:
        adv['cigarettes_per_day'] = None
        adv['cigarette_note'] = None
    return adv

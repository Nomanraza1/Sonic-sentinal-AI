SONIC_CLASSES = [
    "machinery_fault", "glass_breaking", "alarm_siren", "vehicle_horn",
    "animal_sound", "gunshot", "panic_scream", "aggression",
    "person_asking_for_help", "background_noise",
]

DISPLAY_NAMES = {name: name.replace("_", " ").title() for name in SONIC_CLASSES}

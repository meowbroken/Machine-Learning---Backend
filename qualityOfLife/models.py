from django.db import models

class StudentInput(models.Model):
    GENDER_CHOICES = [('Male', 'Male'), ('Female', 'Female'), ('Prefer not to say', 'Prefer not to say')]

    age = models.PositiveIntegerField()
    gender = models.CharField(max_length=25, choices=GENDER_CHOICES)
    grade_level = models.CharField(max_length=25)
    strand = models.CharField(max_length=50)

    internet_access = models.JSONField(help_text="List of selected internet access types")
    devices = models.JSONField(help_text="List of owned digital devices")

    screen_time = models.CharField(max_length=25)
    pre_sleep_use = models.CharField(max_length=50)
    usage_purpose = models.JSONField(help_text="List of selected purposes")
    device_check_freq = models.PositiveSmallIntegerField()
    usage_interrupts_tasks = models.PositiveSmallIntegerField()
    control_level = models.PositiveSmallIntegerField()
    study_distraction = models.PositiveSmallIntegerField()
    wakeup_use = models.BooleanField()
    num_socials = models.PositiveSmallIntegerField()
    attempted_detox = models.BooleanField()
    aware_of_hours = models.BooleanField()
    time_of_day_use = models.CharField(max_length=50)
    sleep_disrupted = models.BooleanField()

    predicted_qol = models.FloatField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

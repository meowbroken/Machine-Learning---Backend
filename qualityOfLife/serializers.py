from rest_framework import serializers
from .models import StudentInput

class StudentInputSerializer(serializers.ModelSerializer):
    class Meta:
        model = StudentInput
        fields = '__all__'
        read_only_fields = ('predicted_qol', 'created_at')

    def validate_age(self, value):
        if not (12 <= value <= 35):
            raise serializers.ValidationError("Age must be between 12 and 35.")
        return value

    def validate_control_level(self, value):
        if not (1 <= value <= 10):
            raise serializers.ValidationError("Control level must be between 1 and 10.")
        return value

    def validate_device_check_freq(self, value):
        if not (1 <= value <= 5):
            raise serializers.ValidationError("Device check frequency must be 1–5.")
        return value

    def validate_usage_interrupts_tasks(self, value):
        if not (1 <= value <= 5):
            raise serializers.ValidationError("Interrupt frequency must be 1–5.")
        return value

    def validate_study_distraction(self, value):
        if not (1 <= value <= 5):
            raise serializers.ValidationError("Study distraction must be 1–5.")
        return value

    def validate_num_socials(self, value):
        if not (0 <= value <= 10):
            raise serializers.ValidationError("Number of social platforms must be between 0 and 10.")
        return value

    def validate_internet_access(self, value):
        if not isinstance(value, list) or not value:
            raise serializers.ValidationError("Internet access must be a non-empty list.")
        return value

    def validate_devices(self, value):
        if not isinstance(value, list) or not value:
            raise serializers.ValidationError("Devices must be a non-empty list.")
        return value

    def validate_usage_purpose(self, value):
        valid_purposes = [
            "Social Media", "Streaming (YouTube, etc.)", "Gaming",
            "Online Shopping", "School Work/ Studying", "Reading / Research",
            "Work-related tasks"
        ]
        if not isinstance(value, list):
            raise serializers.ValidationError("Usage purpose must be a list.")
        for p in value:
            if p not in valid_purposes:
                raise serializers.ValidationError(f"Invalid usage purpose: {p}")
        return value

    def validate_time_of_day_use(self, value):
        valid_values = [
            "Morning (6 AM – 12 PM)", "Afternoon (12 PM – 6 PM)",
            "Evening (6 PM – 10 PM)", "Late Night (10 PM – 3 AM)",
            "Equally throughout the day"
        ]
        if value not in valid_values:
            raise serializers.ValidationError(f"Invalid time of day use: {value}")
        return value

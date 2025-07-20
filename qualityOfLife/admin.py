from django.contrib import admin
from .models import StudentInput

@admin.register(StudentInput)
class StudentInputAdmin(admin.ModelAdmin):
    list_display = ['age', 'gender', 'grade_level', 'predicted_qol', 'created_at']
    readonly_fields = ['predicted_qol', 'created_at']

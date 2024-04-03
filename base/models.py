from django.db import models

class SegmentedImage(models.Model):
    image_path = models.ImageField(upload_to='segmented_images/')
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.image_path.url

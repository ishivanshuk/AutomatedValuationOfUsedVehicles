ALLOWED_EXTENSIONS = ['png', 'jpeg', 'jpg']
UPLOADS_FOLDER = 'uploads/images'

def file_valid(file):
  return '.' in file and \
    file.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS
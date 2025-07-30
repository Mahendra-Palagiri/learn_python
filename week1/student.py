class student:

    def __init__(self, name, score):
        self.name = name
        self.score = score

    def grade_student(self):
        if self.score >= 95:
            print("A+")
        elif self.score > 90:
            print("A")
        elif self.score > 85:
            print("A-")
        else: 
            print("Less than A grade")
        
            
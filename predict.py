from transformers import pipeline

def get_classifier():
    return pipeline("sentiment-analysis")

if __name__ == "__main__":
    classifier = get_classifier()
    result = classifier("I love learning MLOps!")
    print(result)
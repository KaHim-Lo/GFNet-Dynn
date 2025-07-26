from models.gfnet_dynn import TrainingPhase



def switch_training_phase(current_phase):
    if current_phase == TrainingPhase.GATE:
        return TrainingPhase.CLASSIFIER
    elif current_phase == TrainingPhase.CLASSIFIER:
        return TrainingPhase.GATE
    elif current_phase == TrainingPhase.WARMUP:
        return TrainingPhase.GATE


        
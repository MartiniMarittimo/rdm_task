import numpy as np
import torch
import neurogym as ngym


class RandomDotMotion(ngym.TrialEnv):
    
    def __init__(self, deltaT, timing=None, cohs=None, rewards=None, sigma=1.0, dim_ring=2):
        
        super().__init__(dt=deltaT)
            
        self.timing = {'fixation': 750, 'stimulus': np.arange(100, 1600, 100), 'delay': 0, 'decision': 500}
        if timing:
            self.timing.update(timing) 
            #modo di updatare un dictionary: o aggiunge gli elementi in fondo o sostituisce quelli con la stessa chiave
        
        if cohs is None:
            self.coherences = [-16, -8, -4, -2, -1, 1, 2, 4, 8, 16]
        else:
            self.coherences = cohs
            
        self.rewards = {'abort': -0.1, 'correct': +1., 'fail': 0.}
        if rewards:
            self.rewards.update(rewards)
            
        #self.abort = False
        
        #self.SCALE = 3.2
        
####################################################################################
        self.theta = np.linspace(0, 2*np.pi, dim_ring+1)[:-1]
        self.choices = np.arange(dim_ring)

        name = {'fixation': 0, 'stimulus': range(1, dim_ring+1)}
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(1+dim_ring,), dtype=np.float32, name=name)
        name = {'fixation': 0, 'choice': range(1, dim_ring+1)}
        self.action_space = spaces.Discrete(1+dim_ring, name=name)
####################################################################################
        
    
    def _new_trial(self, **kwargs):
        """
        new_trial() is called when a trial ends to generate the next trial.
        The following variables are created:
            durations, which stores the duration of the different periods (in the case of perceptualDecisionMaking: 
            fixation, stimulus and decision periods)
            ground truth: correct response for the trial
            coh: stimulus coherence (evidence) for the trial
            obs: observation
        """
        # Trial info
        trial = {
            'ground_truth': self.rng.choice(self.choices),
            'coh': self.rng.choice(self.coherences),  #self.rng = np.random.RandomState()
        }
        trial.update(kwargs) #???

        ground_truth = trial['ground_truth']        
        coh = trial['coh']
        stim_theta = self.theta[ground_truth]

        # Periods
        self.add_period(['fixation', 'stimulus', 'delay', 'decision'])

        # Observations
        self.add_ob(1, period=['fixation', 'stimulus', 'delay'], where='fixation')
        stim = np.cos(self.theta - stim_theta) * (coh/200) + 0.5
        self.add_ob(stim, 'stimulus', where='stimulus')
        self.add_randn(0, self.sigma, 'stimulus', where='stimulus')

        # Ground truth
        self.set_groundtruth(ground_truth, period='decision', where='choice')

        return trial
        
        
    def _step(self, action):
        """
        _step receives an action and returns:
            a new observation, obs
            reward associated with the action, reward
            a boolean variable indicating whether the experiment has end, done
            a dictionary with extra information:
                ground truth correct response, info['gt']
                boolean indicating the end of the trial, info['new_trial']
        """
        new_trial = False
        
        reward = 0
        gt = self.gt_now #ground_truth
        
        if self.in_period('fixation'):
            if action != 0:  # action = 0 means fixating
                new_trial = False
                reward += self.rewards['abort']
                
        elif self.in_period('decision'):
            if action != 0:
                new_trial = True
                if action == gt:
                    reward += self.rewards['correct']
                    self.performance = 1
                else:
                    reward += self.rewards['fail']

        return self.ob_now, reward, False, {'new_trial': new_trial, 'gt': gt}
        
        
   
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

"""
def setup():
    """
    #Redefine global variables for the stimuli
"""
    global fixation_discrete, stimulus_discrete, decision_discrete
    #global index_response
    
    fixation_discrete = int(fixation / deltaT) #15
    stimulus_discrete = stimulus // deltaT #40
    decision_discrete = int(decision / deltaT) #7
    
    #index_response = fixation_discrete + stimulus_discrete + delay_discrete


setup()


def generate_rdm_data(num_trials, coherences=None, std=3e-2, fraction_validation_trials=.2, fraction_catch_trials=0.2):
    
     # coh=dx-sx

    zero_coherence_trials = []
    values = []
    
    inputs = torch.zeros((num_trials, 275//2, 3), dtype=torch.float32) # 0:fix, 1:dx, 2:sx
    targets = torch.zeros((num_trials, 275//2, 3), dtype=torch.float32) # 0:fix, 1:dx, 2:sx
    mask = torch.ones((num_trials, 275//2, 1), dtype=torch.float32)
    
    for i in range(num_trials):
        
        trial_values = []
        
        stimulus_duration = int(np.random.choice(stimulus_discrete))
        stimulus_end = fixation_discrete + stimulus_duration
        max_total_duration = fixation_discrete + stimulus_duration + decision_discrete
        
        inputs[i, :stimulus_end, 0] = 1
        targets[i, :stimulus_end, 0] = 1 

        if np.random.rand() > fraction_catch_trials:
            zero_coherence_trials.append(0)
            coh_current = np.random.choice(coherences)
            trial_values.append(coh_current)
            if coh_current > 0: # coh=dx-sx
                inputs[i, fixation_discrete:stimulus_end, 1] += std * torch.randn(stimulus_duration) +\
                (coh_current+1) * SCALE / 100
                inputs[i, fixation_discrete:stimulus_end, 2] += std * torch.randn(stimulus_duration) +\
                (1) * SCALE / 100
                targets[i, stimulus_end:, 1] = hi 
            else: 
                inputs[i, fixation_discrete:stimulus_end, 1] += std * torch.randn(stimulus_duration) +\
                (1) * SCALE / 100
                inputs[i, fixation_discrete:stimulus_end, 2] += std * torch.randn(stimulus_duration) +\
                (-coh_current+1) * SCALE / 100
                targets[i, stimulus_end:, 2] = hi 
        
        else:
            zero_coherence_trials.append(1)
            trial_values.append(0)
            
        mask[i, max_total_duration:, 0] = 0
        trial_values.append(fixation_discrete)
        trial_values.append(stimulus_duration+fixation_discrete) 
        trial_values.append(max_total_duration)
        values.append(trial_values)

    # Split the generated dataset into a training set and a validation set
    split_at = num_trials - int(num_trials * fraction_validation_trials)
    inputs_train, inputs_val = inputs[:split_at], inputs[split_at:]
    targets_train, targets_val = targets[:split_at], targets[split_at:]
    mask_train, mask_val = mask[:split_at], mask[split_at:]
    values_train, values_valid = values[:split_at], values[split_at:]

    return inputs_train, targets_train, mask_train, inputs_val, targets_val, mask_val, values_train, values_valid
"""
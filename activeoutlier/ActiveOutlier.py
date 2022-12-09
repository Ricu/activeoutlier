import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


from scipy import stats
from sklearn.tree import DecisionTreeClassifier 

class ActiveOutlier:
  def __init__(self, background_distr='uniform',
               base_learner=DecisionTreeClassifier,
               base_learner_params={},
               probabilistic=False
               ):
    
    self.base_learner = base_learner
    self.base_learner_params = base_learner_params
    self.background_distr = background_distr
    self.probabilistic = probabilistic
    self.syn_color = sns.color_palette()[0]
    self.real_color = sns.color_palette()[1]

  def fit(self,
          X_real,
          n_steps=10,
          verbose=1,
          random_state=None,
          h0_type=0,
          starting_fraction=0.5,
          sample_fraction=None
          ):
    
        self.real_samples, self.n_features = X_real.shape
        self.n_samples = self.real_samples * 2
        self.n_steps = n_steps
        self.clf_list = []
        self.clf_weights = np.zeros(n_steps)
        self.errors = np.zeros(n_steps)
        self.margin_history = np.zeros((self.n_samples,n_steps))
        self.sampling_prob_history = np.zeros((self.n_samples,n_steps))
        self.scaling_history = np.zeros(n_steps)
        self.sampling_history = np.zeros((self.n_samples,n_steps))
        self.train_preds = np.zeros((self.n_samples,n_steps+1))

        # create the training dataset by creating and appending artificial datapoints
        # according to the given background distribution
        diff = X_real.max()-X_real.min()
        prng = np.random.default_rng(random_state)
        self.real_range = diff

        if self.background_distr=='uniform':
          artificial_samples = prng.uniform(low=X_real.min()-0.1*diff,
                                            high=X_real.max()+0.1*diff,
                                            size=X_real.shape)
        elif self.background_distr=='normal':
          artificial_samples = prng.normal(loc=X_real.mean(),
                                           scale=1.1*X_real.std(),
                                           size=X_real.shape)
        else:
          ValueError('Invalid background distribution!')
          
        X = pd.DataFrame(np.concatenate([X_real.values,artificial_samples],
                                        axis=0),
                         columns = X_real.columns)
        y = np.zeros(self.n_samples)
        y[:self.real_samples] = 1
        y = pd.Series(y, name = 'y')
        self.labels = y
        self.datapoints = X


        # h_0 prediction
        if h0_type == 'random':
          # variant 1: random predictions for h_0
          self.train_preds[:,0] = prng.integers(0,2,size=self.n_samples)
        elif h0_type == 'y_true':
          # variant 2: use y_true
          # in step 2: will focus on the wrongly classified samples
          # (will focus on synthetic samples in dense real regions)
          self.train_preds[:,0] = y
        elif h0_type == 'y_true_inv':
          # variant 3: take inverse of y_true
          # in step 2: will focus on the correctly classified samples
          # (will focus on all but synthetic samples in dense real regions)
          self.train_preds[:,0] = 1-y
        elif h0_type == 'focus_potential_inliers':
          # variant 4: choose all outliers (class 0)
          # in step 2: focus on samples classified as 1
          self.train_preds[:,0] = 0
        elif h0_type == 'focus_potential_outliers':
          # variant 5: choose all inliers (class 1)
          # in step 2: focus on samples classified as 0
          self.train_preds[:,0] = 1


        for step in range(n_steps):
          # calculate margin for each sample
          pre_margin = self.train_preds[:,:step+1].copy().reshape(self.n_samples,step+1)
          pre_margin[pre_margin<0.5] = -(1-pre_margin[pre_margin<0.5])
          margin = abs(pre_margin.sum(axis=1))
          self.margin_history[:,step] = margin

          # calculate sampling probability for each sample
          sample_prob = 1 - stats.norm.cdf(x=(step+1+margin)/2,
                                           loc=(step+1)/2,
                                           scale=np.sqrt(step+1)/2)
          self.sampling_prob_history[:,step] = sample_prob
          
          # normalizing factor the amount of samples
          scaling = 1
          if sample_fraction is not None:
            scaling = self.n_samples*sample_fraction / sample_prob.sum()
          self.scaling_history[step] = scaling

          # sampling the datapoints
          sample_ids = np.where(scaling*sample_prob >= prng.random(self.n_samples))[0]

          self.sampling_history[sample_ids,step] = 1
          # new classifier
          X_sampled = X.iloc[sample_ids,]
          y_sampled = y.iloc[sample_ids,]
          self.clf_list.append(self.base_learner(**self.base_learner_params,
                                                 random_state=random_state)
                                                 )
          self.clf_list[-1].fit(X=X_sampled,y=y_sampled)

          # calculate predictions for new classifier
          if self.probabilistic:
            self.train_preds[:,step+1] = self.clf_list[-1].predict_proba(X)[:,1]
          else:
            self.train_preds[:,step+1] = self.clf_list[-1].predict(X)
          
          # calculate weights
          # print('y_sampled sum {}'.format(y_sampled.sum()))
          # print('y_predicted sum {}'.format(self.clf_list[-1].predict(X_sampled).sum()))
          error = 1-self.clf_list[-1].score(X_sampled,y_sampled)
          self.errors[step] = error
          eps = 10**(-8)
          self.clf_weights[step] = np.log((1-error+eps)/(error+eps))

          if verbose != 0:
            print('step={}, n_sampled={}, error={:.3f}, weight={:.3f}'.format(step,sample_ids.shape,error,self.clf_weights[step]))

  def predict(self,
              X,
              step = None,
              threshold = 0.5
              ):
    if step is None:
      step = self.n_steps

    if self.probabilistic:
      raw_predictions = np.array(list(map(lambda clf: clf.predict_proba(X)[:, 1],self.clf_list[:step]))).T
    else:
      raw_predictions = np.array(list(map(lambda clf: clf.predict(X),self.clf_list[:step]))).T
      # outputs are supposed to be -1,1 for prediction s.t. weighting makes sense
      raw_predictions = raw_predictions*2-1

    weighted_prediction_sums  = (raw_predictions*self.clf_weights[:step]).sum(axis=1)
    y_pred = np.sign(weighted_prediction_sums-threshold*self.clf_weights[:step].sum())
    y_pred = np.round((y_pred+1)/2)
    return y_pred

  def predict_proba(self,
                    X,
                    step = None,
                    threshold = 0.5
                    ):
    if step is None:
      step = self.n_steps

    if self.probabilistic:
      raw_predictions = np.array(list(map(lambda clf: clf.predict_proba(X)[:, 1],self.clf_list[:step]))).T
    else:
      raw_predictions = np.array(list(map(lambda clf: clf.predict(X),self.clf_list[:step]))).T

    weighted_prediction_sums  = (raw_predictions*self.clf_weights[:step]).sum(axis=1)

    y_pred = np.zeros((X.shape[0],2))
    y_pred[:,1] = weighted_prediction_sums/(self.clf_weights[:step].sum())
    y_pred[:,0] = 1-y_pred[:,1]

    return y_pred

  def plot_sampling_probability(self,
                                feature_x,
                                feature_y,
                                step = None,
                                scaled=False,
                                ax=None
                                ):
    if step is None:
      step = self.n_steps

    samples = self.datapoints.copy()
    if scaled:
      samp_txt = 'scaled_sampling_probability'
      samples[samp_txt] = (self.scaling_history[step] * self.sampling_prob_history[:,step]).round(3)
    else:
      samp_txt = 'sampling_probability'
      samples[samp_txt] = self.sampling_prob_history[:,step].round(3)

    samples['ground_truth'] = self.labels


    ax = sns.scatterplot(data=samples,
                         x = feature_x,
                         y=feature_y,
                         size = samples[samp_txt],
                         sizes = (60,240),
                         hue = 'ground_truth',
                         alpha = 0.7,
                         ax=ax
                         )
    ax.set_title('Plot of {} in step={} (python={})'.format(samp_txt.replace('_',' '),step+1,step),
                 {'fontweight':'bold'})
    return ax
  
  def plot_samples_chosen(self,
                          feature_x,
                          feature_y,
                          step=None,
                          ax=None
                          ):
    


    if step is None:
      step = self.n_steps

    samples = self.datapoints.copy()
    samples['ground_truth'] = self.labels
    samples['chosen_samples'] = self.sampling_history[:,step]

    ax = sns.scatterplot(data=samples[samples['chosen_samples']==1],
                         x=feature_x,
                         y=feature_y,
                         hue='ground_truth',
                         alpha=0.7,
                         s=60,
                         ax=ax
                         )
    ax.set_title('Plot of chosen samples in step={} (python={})'.format(step+1,step),
                 {'fontweight':'bold'})
    return ax

  def plot_data_distributions(self,
                              feature_x,
                              feature_y,
                              kind='marginal_plot',
                              plot_real=True,
                              plot_syn=True
                              ):
    data = self.datapoints
    labels = self.labels
    sns.set_style('white')



    if kind == 'scatter_plot':
      ax_joint = sns.scatterplot(data=[])
    elif kind == 'marginal_plot':
      g = sns.JointGrid(space=0,
                        height = 8,
                        xlim=(data[feature_x].min()-0.5,data[feature_x].max()+0.5),
                        ylim=(data[feature_y].min()-0.5,data[feature_y].max()+0.5))
      ax_joint = g.ax_joint
    else:
      raise ValueError('Not a valid plot type')

    if plot_syn:
      sns.scatterplot(x=data.loc[labels==0,feature_x],
                      y=data.loc[labels==0,feature_y],
                      ax=ax_joint,
                      s=30,
                      label='artificial samples (from $B$)',
                      alpha = 0.7,
                      facecolor = self.syn_color,
                      edgecolor = 'black'
                      )
      
      # plot marginals
      if kind == 'marginal_plot':
        sns.kdeplot(x=data.loc[labels==0,feature_x],
                    ax = g.ax_marg_x,
                    fill = True,
                    color = self.syn_color
                    )
        sns.kdeplot(y=data.loc[labels==0,feature_y],
                    ax = g.ax_marg_y,
                    fill = True,
                    color = self.syn_color
                    )
    if plot_real:
      sns.scatterplot(x=data.loc[labels==1,feature_x],
                      y=data.loc[labels==1,feature_y],
                      ax=ax_joint,
                      s=60,
                      label='real samples (from $U$)',
                      alpha=0.9,
                      facecolor = self.real_color,
                      edgecolor = 'black'
                      )

      # plot marginals
      if kind == 'marginal_plot':
        sns.kdeplot(x=data.loc[labels==1,feature_x],
                    ax = g.ax_marg_x,
                    fill = True,
                    color = self.real_color
                    )
        sns.kdeplot(y=data.loc[labels==1,feature_y],
                    ax = g.ax_marg_y,
                    fill = True,
                    color = self.real_color
                  )
    
    # move legend
    sns.move_legend(ax_joint, 'upper right')

    return ax_joint    
  
  def plot_decision_boundary(self,
                             feature_x,
                             feature_y,
                             threshold=0.5,
                             step=None,
                             n=201,
                             ax=None,
                             plot_data=False
                             ):
    real_samples = self.labels==1
    x_min = self.datapoints.loc[real_samples,feature_x].min()-0.1*self.real_range[feature_x]
    x_max = self.datapoints.loc[real_samples,feature_x].max()+0.1*self.real_range[feature_x]
    y_min = self.datapoints.loc[real_samples,feature_y].min()-0.1*self.real_range[feature_y]
    y_max = self.datapoints.loc[real_samples,feature_y].max()+0.1*self.real_range[feature_y]

    xx,yy = np.meshgrid(np.linspace(x_min,x_max, n),
                        np.linspace(y_min,y_max, n))
    grid = pd.DataFrame(np.concatenate((xx.flatten().reshape(-1,1),yy.flatten().reshape(-1,1)),axis = 1),
                          columns = [feature_x,feature_y])
    pred = self.predict(grid,
                        threshold=threshold,
                        step=step)
    ax = sns.histplot(data=grid,
                      x=feature_x,
                      y=feature_y,
                      weights=pred,
                      bins=(np.linspace(x_min,y_max, n),np.linspace(y_min,y_max, n)),
                      color='g',
                      ax=ax
                      )
    font = {
      'weight': 'normal',
      'size'  :  20,
      'color': 'lightgray'
    }
    label = ax.text(0.95, 0.2,
                    'threshold: {}'.format(np.round(threshold,2)),
                    horizontalalignment='right',
                    verticalalignment='top',
                    transform=ax.transAxes,
                    fontdict=font)
    ax.set_title('Decision boundary plot with marked outliers',
                 {'fontweight':'bold'})
    if plot_data:
      ax = self.plot_outliers(feature_x,
                              feature_y,
                              threshold=threshold,
                              ax=ax
                              )


    return ax

  def plot_outliers(self,
                    feature_x,
                    feature_y,
                    threshold=0.5,
                    plot_syn=False,
                    ax=None
                    ):
    

    data = self.datapoints
    labels = self.labels
    if not plot_syn:
      data = data.loc[labels==1,]
      labels = labels.loc[labels==1,]
      
    sns.set_style('white')

    pred = self.predict(data)

    outliers = (labels==1)&(pred==0)
    inliers = (labels==1)&(pred==1)

    ax = sns.scatterplot(x=data.loc[inliers,feature_x],
                          y=data.loc[inliers,feature_y],
                          ax=ax,
                          s=30,
                          label='real samples (predicted inliers)',
                          alpha=0.9,
                          facecolor = self.real_color,
                          edgecolor = 'black'
                          )
    
    sns.scatterplot(x=data.loc[outliers,feature_x],
                    y=data.loc[outliers,feature_y],
                    ax=ax,
                    s=60,
                    label='real samples (predicted outliers)',
                    alpha=0.9,
                    facecolor = 'red',
                    edgecolor = 'black'
                    )

    if plot_syn:
      sns.scatterplot(x=data.loc[labels==0,feature_x],
                      y=data.loc[labels==0,feature_y],
                      ax=ax,
                      s=30,
                      label='artificial samples (from $B$)',
                      alpha = 0.7,
                      facecolor = self.syn_color,
                      edgecolor = 'black'
                      )


    # move legend
    sns.move_legend(ax, 'upper right')

    return ax 
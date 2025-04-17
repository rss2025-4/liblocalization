##################
 install or upate
##################

.. code:: text

   # unisntall existing if updating
   pip uninstall liblocalization -y

   # latest pip install --upgrade
   git+https://github.com/rss2025-4/liblocalization.git # or from branch
   pip install --upgrade
   git+https://github.com/rss2025-4/liblocalization.git@branch-xxxx # or
   from revision pip install --upgrade
   git+https://github.com/rss2025-4/liblocalization.git@xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

###########
 interface
###########

.. currentmodule:: liblocalization

.. autoclass:: LocalizationBase

.. autoclass:: localization_params

#############
 controllers
#############

.. autofunction:: deterministic_motion_tracker

.. autofunction:: particles_model

.. autoclass:: particles_params

#########
 example
#########

.. autoclass:: ExampleSimNode
   :no-members:

.. autofunction:: examplemain

.. autofunction:: examplemain2

############
 parameters
############

.. autofunction:: liblocalization.motion.motion_model

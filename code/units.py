##########################################
# Everywhere where pint is used in this  #
# project, it should be imported from    #
# here. Otherwise, an issue will raise   #
# since different UnitRegistry() !       #
#                                        #
# Just do: from units import u           #
#    Then you can use it e.g., u.ms      #
##########################################

import pint
u = pint.UnitRegistry()
pint.set_application_registry(u)
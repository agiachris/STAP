(define (problem long_horizon_0)
	(:domain symbolic_workspace)
	(:objects
        rack - receptacle
		hook - tool
        blue_box - box
	)
	(:init
		(on rack table)
        (on hook table)
		(on blue_box table)
	)
	(:goal (and
        (under blue_box rack)
	))
)

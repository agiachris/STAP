(define (problem long_horizon_0)
	(:domain geometric_workspace)
	(:objects
        rack - receptacle
		hook - tool
        blue_box - box
	)
	(:init
		(on rack table)
        (on hook table)
		(on blue_box table)
        ; Geometric facts
        (inworkspace table)
        (inworkspace hook)
        (incollisionzone blue_box)
        (beyondworkspace rack)
        (nonblocking rack hook)
	)
	(:goal (and
        (under blue_box rack)
	))
)

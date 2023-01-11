(define (problem real_world_0)
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
        (inworkspace rack)
        (inworkspace hook)
        (beyondworkspace blue_box)
	)
	(:goal (and
        (on blue_box rack)
	))
)

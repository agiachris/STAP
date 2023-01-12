(define (problem real_world_0)
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
        (on blue_box rack)
	))
)

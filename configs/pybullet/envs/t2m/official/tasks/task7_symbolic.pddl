(define (problem lifted_1)
	(:domain symbolic_workspace)
	(:objects
        rack - receptacle
        hook - tool
        cyan_box - box
        blue_box - box
	)
	(:init
		(on rack table)
        (on hook table)
		(on cyan_box table)
		(on blue_box table)
	)
	(:goal (and
        (on blue_box rack)
    ))
)

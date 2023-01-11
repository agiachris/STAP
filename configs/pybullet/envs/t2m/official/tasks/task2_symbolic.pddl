(define (problem structured_language_2)
	(:domain symbolic_workspace)
	(:objects
        rack - receptacle
		hook - tool
        yellow_box - box
        cyan_box - box
	)
	(:init
		(on rack table)
        (on hook table)
		(on yellow_box table)
		(on cyan_box table)
	)
	(:goal (and
        (under yellow_box rack)
	))
)

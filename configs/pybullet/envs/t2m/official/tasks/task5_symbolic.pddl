(define (problem lifted_0)
	(:domain symbolic_workspace)
	(:objects
        rack - receptacle
        hook - tool
        red_box - box
        yellow_box - box
        cyan_box - box
	)
	(:init
		(on rack table)
        (on hook table)
		(on red_box table)
		(on yellow_box table)
		(on cyan_box table)
	)
	(:goal (or
        (and (on red_box rack))
        (and (on yellow_box rack))
        (and (on cyan_box rack))
        (and (on red_box rack)
             (on yellow_box rack))
        (and (on red_box rack)
             (on cyan_box rack))
        (and (on yellow_box rack)
             (on cyan_box rack))
        (and (on red_box rack)
             (on yellow_box rack)
             (on cyan_box rack))
    ))
)
(define (domain workspace)
	(:requirements :strips :typing :equality :universal-preconditions :negative-preconditions :conditional-effects)
	(:types
		physobj - object
		unmovable - physobj
		movable - physobj
		tool - movable
		box - movable
		receptacle - unmovable
	)
	(:constants table - unmovable)
	(:predicates
		(inhand ?a - movable)
		(on ?a - movable ?b - unmovable)
		(inworkspace ?a - physobj)
		(beyondworkspace ?a - physobj)
		(under ?a - movable ?b - receptacle)
		(aligned ?a - physobj)
		(poslimit ?a - physobj)
	)
	(:action pick
		:parameters (?a - physobj)
		:precondition (and
            (or
                ; 1) (state) Movable inhand, 
                ; 1a)       (action) Pick physobj. 
                (exists (?b - movable) (inhand ?b))
                
                ; 2) (state) Nothing inhand, 
                ; 2a)       (action) Pick unmovable.
                (and 
                    (forall (?b - movable) (not (inhand ?b)))
                    (exists (?b - unmovable) (= ?a ?b))
                )
            )
        )
		:effect (and )
	)
	(:action place
		:parameters (?a - physobj ?b - physobj)
		:precondition (and
			(not (= ?a ?b))
            (or
                ; 1) (state) Movable inhand.
                ; 1a)       (action) inhand(?a), place on another movable.
                (and (inhand ?a) (exists (?c - movable) (= ?b ?c)))
                ; 1b)       (action) (not (inhand ?a)), place on anything.
                (exists (?c - movable) (and (inhand ?c) (not (= ?a ?c))))

                ; 2) (state) Nothing inhand. 
                ; 2a)       (action) Place anything on anything.
                (forall (?c - movable) (not (inhand ?c)))
            )
		)
		:effect (and )
    )
	(:action pull
		:parameters (?a - physobj ?b - physobj)
		:precondition (and
			(not (= ?a ?b))
            (or 
                ; 1) (state) Movable inhand.
                ; 1a)       (state) inhand(?b), b is a tool.
                (and (inhand ?b) (exists (?c - tool) (= ?b ?c))
                    (or
                ; 1ax)              (action) Pull movable on rack with tool.
                        (and 
                            (exists (?c - movable) (= ?a ?c))
                            (exists (?c - receptacle) (on ?a ?c))
                        )
                ; 1ay)              (action) Pull unmovable with tool.
                        (exists (?c - unmovable) (= ?a ?c))
                    )
                )
                
                ; 1b)       (state) inhand(?b), b is a box.
                ; 1bx)              (action) Pull anything with box.
                (and (inhand ?b) (exists (?c - box) (= ?b ?c)))

                ; 1c)       (state) (not (inhand ?b)), 
                ; 1cx)              (action) Pull anything with anything.
                (not (inhand ?b))
                
                ; 2) (state) Nothing inhand.
                ; 2a)       (action) Pull anything with anything.
                (forall (?c - movable) (not (inhand ?c)))
            )
		)
		:effect (and )
	)
    (:action push
        :parameters (?a - physobj ?b - physobj ?c - receptacle)
        :precondition (and
            (not (= ?a ?b))
            (not (= ?a ?c))
            (not (= ?b ?c))
            (or
                ; 1) (state) Movable inhand
                ; 1a)       (state) inhand(?b), b is a tool
                (and (inhand ?b) (exists (?d - tool) (= ?b ?d))
                    (or
                ; 1ax)              (action) Push movable on rack with tool.
                        (and
                            (exists (?d - movable) (= ?a ?d))
                            (exists (?d - receptacle) (on ?a ?d))
                        )
                ; 1ay)              (action) Push ummovable with tool.
                        (exists (?d - unmovable) (= ?a ?d))
                    )
                )
                ; 1b)       (state) inhand(?b), b is a box
                ; 1bx)              (action) Push anything with box.
                (and (inhand ?b) (exists (?d - box) (= ?b ?d)))
                
                ; 1c)       (state) (not (inhand ?b))
                ; 1cx)              (action) Push anything with anything.
                (not (inhand ?b))

                ; 2) (state) Nothing inhand.
                ; 2a)       (action) Push anything with anything.
                (forall (?d - movable) (not (inhand ?d)))
            )
        )
        :effect (and )
    )
)
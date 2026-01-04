from manimlib import *
import numpy as np

from manim_slides import Slide

class CRRTree(Slide):
    def setup(self):
        self.T = 0.5
        self.n = 3
        self.delta_t = self.T / self.n
        self.riskfree_rate = 0.06
        self.asset_volatility = 0.2
        self.asset_price = 100
        self.strike_price = 95
        self.tilt = 0
        self.discrete_rate_factor = np.exp(self.riskfree_rate * self.delta_t)
        
        self.u = np.exp(self.asset_volatility * np.sqrt(self.delta_t) + self.tilt * self.asset_volatility**2 * self.delta_t)
        self.d = np.exp(-self.asset_volatility * np.sqrt(self.delta_t) + self.tilt * self.asset_volatility**2 * self.delta_t)

        self.y_min = self.asset_price * self.d**self.n
        self.y_max = self.asset_price * self.u**self.n

    def S(self, i, j):
        return self.asset_price * self.u**j * self.d**(i-j)

    def construct(self):
        self.setup()

        price_range = (np.log(self.y_min), np.log(self.y_max), (np.log(self.y_max) - np.log(self.y_min)) / self.n)
        t_range = (0, self.T, self.delta_t)

        t_axis = NumberLine(
            x_range=t_range,
            width=10,
            decimal_number_config={
                'font_size': 20,
                'num_decimal_places': 2,
            },
        )
        t_axis.add_numbers()
        t_axis.to_edge(DOWN)
        self.play(ShowCreation(t_axis))

        T_label = Text('T =', font_size=22)
        T_label.next_to(t_axis.get_number_mobject(self.T), LEFT)

        self.play(Write(T_label))
        self.next_slide()

        delta_t_segment = Line(t_axis.n2p(0), t_axis.n2p(self.delta_t))
        delta_t_segment.set_stroke(color=YELLOW_A)
        brace = Brace(delta_t_segment, UP)
        brace_label = Tex(r'\Delta t = \frac{T}{n}', font_size=32)
        brace_label.next_to(brace, UP)
        self.play(Write(brace), Write(brace_label))

        self.next_slide()
        self.play(FadeOut(T_label), FadeOut(brace), FadeOut(brace_label))

        price_axis = NumberLine(
            x_range=price_range,
            width=6,
            include_ticks=False,
            decimal_number_config={
                'font_size': 20,
                'num_decimal_places': 2,
            },
            line_to_number_direction=LEFT,
        )
        price_axis.rotate(90 * DEG, about_point=ORIGIN)

        prices = [
            self.S(i, j) for i in (self.n - 1, self.n) for j in range(i+1)
        ]
        prices.sort()

        ticks = VGroup()
        for price in prices:
            ticks.add(price_axis.get_tick(np.log(price), price_axis.tick_size))
        price_axis.add(ticks)

        for price, tick in zip(prices, ticks):
            label = DecimalNumber(price, font_size=20, num_decimal_places=2)
            always(label.next_to, tick, LEFT, SMALL_BUFF)
            price_axis.add(label)

        # price_axis.add_numbers(prices)
        price_axis.to_edge(LEFT)
        self.play(ShowCreation(price_axis))

        def c2p(t, S):
            t_aligned = t_axis.n2p(t)
            t_aligned[1] = price_axis.n2p(np.log(S))[1]
            return t_aligned
            
        self.next_slide()

        NODE_COLOR = YELLOW_B
        EDGE_COLOR = YELLOW_E

        S_0 = Dot(c2p(0, self.asset_price), fill_color=NODE_COLOR)
        S_0.set_z_index(1)
        label = Tex('S_0 = 100', font_size=20)
        label.next_to(S_0, DOWN)

        self.play(GrowFromCenter(S_0), Write(label))

        self.next_slide()

        self.play(FadeOut(label))

        # add formulas
        u_formula = Tex(r'u = e^{\sigma \sqrt{\Delta t}} \approx ' + str(round(self.u, 3)), font_size=32)
        d_formula = Tex(r'd = e^{-\sigma \sqrt{\Delta t}} \approx ' + str(round(self.d, 3)), font_size=32)
        formulas = VGroup(u_formula, d_formula).arrange(DOWN)
        formulas.next_to(c2p(0, self.y_max), RIGHT)
        self.play(Write(formulas))

        self.next_slide()

        S_layers: VGroup[VGroup[Dot]] = VGroup()
        S_layers.add(VGroup(S_0))

        for layer in range(1, self.n+1):
            layer_group = VGroup()
            prev_layer = S_layers[layer-1]
            lower_child = Dot(c2p(layer * self.delta_t, self.S(layer, 0)), fill_color=NODE_COLOR)
            lower_child.set_z_index(layer)
            layer_group.add(lower_child)

            for up_moves in range(len(prev_layer)):
                node = prev_layer[up_moves]
                u_pos = c2p(layer * self.delta_t, self.S(layer, up_moves+1))
                d_pos = c2p(layer * self.delta_t, self.S(layer, up_moves))
                d_edge = DashedLine(node.get_center(), d_pos)
                d_edge.set_stroke(color=EDGE_COLOR, width=2)
                u_edge = DashedLine(node.get_center(), u_pos)
                u_edge.set_stroke(color=EDGE_COLOR, width=2)

                upper_child = Dot(u_pos, fill_color=NODE_COLOR)
                upper_child.set_z_index(layer)
                layer_group.add(upper_child)

                if layer == 1:
                    u_label = Tex(f'\\times u', font_size=32)
                    d_label = Tex(f'\\times d', font_size=32)

                    u_label.next_to(u_edge, UP)
                    d_label.next_to(d_edge, DOWN)

                    self.play(Write(d_label), ShowCreation(d_edge), GrowFromCenter(lower_child))
                    self.next_slide()
                    self.play(Write(u_label), ShowCreation(u_edge), GrowFromCenter(upper_child))
                    self.next_slide()
                    self.play(FadeOut(u_label), FadeOut(d_label))
                else:
                    if up_moves == 0:
                        self.play(ShowCreation(d_edge), ShowCreation(u_edge), GrowFromCenter(lower_child), GrowFromCenter(upper_child))
                    else:
                        self.play(ShowCreation(d_edge), ShowCreation(u_edge), GrowFromCenter(upper_child))
            
            S_layers.add(layer_group)
            if layer > 1:
                self.next_slide()


class FBTree(Slide):
    def setup(self):
        self.T = 0.5
        self.n = 8
        self.delta_t = self.T / self.n
        self.riskfree_rate = 0.06
        self.asset_volatility = 0.2
        self.asset_price = 100
        self.strike_price = 95
        self.tilt = ValueTracker(0)
        self.discrete_rate_factor = np.exp(self.riskfree_rate * self.delta_t)
        
        # self.u = np.exp(self.asset_volatility * np.sqrt(self.delta_t) + self.tilt * self.asset_volatility**2 * self.delta_t)
        self.u0 = np.exp(self.asset_volatility * np.sqrt(self.delta_t))
        self.u = always_redraw(ExponentialValueTracker, self.asset_volatility * np.sqrt(self.delta_t) + self.tilt.get_value() * self.asset_volatility**2 * self.delta_t)
        # self.d = np.exp(-self.asset_volatility * np.sqrt(self.delta_t) + self.tilt * self.asset_volatility**2 * self.delta_t)
        self.d0 = np.exp(-self.asset_volatility * np.sqrt(self.delta_t))
        self.d = always_redraw(ExponentialValueTracker, -self.asset_volatility * np.sqrt(self.delta_t) + self.tilt.get_value() * self.asset_volatility**2 * self.delta_t)

        scaling_factor = 1.2
        self.y_min = self.asset_price * self.d0**self.n / scaling_factor
        self.y_max = self.asset_price * self.u0**self.n * scaling_factor

        self.NODE_COLOR = YELLOW_B
        self.EDGE_COLOR = YELLOW_E

    def c2p(self, t, S):
        t_aligned = self.t_axis.n2p(t)
        t_aligned[1] = self.price_axis.n2p(np.log(S))[1]
        return t_aligned

    def constructTree(self, tilt):
        u = np.exp(self.asset_volatility * np.sqrt(self.delta_t) + tilt * self.asset_volatility**2 * self.delta_t)
        d = np.exp(-self.asset_volatility * np.sqrt(self.delta_t) + tilt * self.asset_volatility**2 * self.delta_t)

        def S(i, j):
            return self.asset_price * u**j * d**(i-j)

        S_0 = Dot(self.c2p(0, self.asset_price), fill_color=self.NODE_COLOR)
        S_0.set_z_index(1)

        tree = VGroup(S_0)

        S_layers: VGroup[VGroup[Dot]] = VGroup()
        S_layers.add(VGroup(S_0))

        for layer in range(1, self.n+1):
            layer_group = VGroup()
            prev_layer = S_layers[layer-1]
            lower_child = Dot(self.c2p(layer * self.delta_t, S(layer, 0)), fill_color=self.NODE_COLOR)
            lower_child.set_z_index(layer)
            layer_group.add(lower_child)

            for up_moves in range(len(prev_layer)):
                node = prev_layer[up_moves]
                u_pos = self.c2p(layer * self.delta_t, S(layer, up_moves+1))
                d_pos = self.c2p(layer * self.delta_t, S(layer, up_moves))
                d_edge = Line(node.get_center(), d_pos)
                d_edge.set_stroke(color=self.EDGE_COLOR, width=1)
                u_edge = Line(node.get_center(), u_pos)
                u_edge.set_stroke(color=self.EDGE_COLOR, width=1)

                u_edge.set_z_index(-1)
                d_edge.set_z_index(-1)

                upper_child = Dot(u_pos, fill_color=self.NODE_COLOR)
                upper_child.set_z_index(layer)
                layer_group.add(upper_child)
            
                if up_moves == 0:
                    tree.add(d_edge, lower_child, u_edge, upper_child)
                else:
                    tree.add(d_edge, u_edge, upper_child)

            S_layers.add(layer_group)
        return tree

    def S(self, i, j):
        return self.asset_price * self.u.get_value()**j * self.d.get_value()**(i-j)

    def construct(self):
        self.setup()

        price_range = (np.log(self.y_min), np.log(self.y_max), (np.log(self.y_max) - np.log(self.y_min)) / self.n)
        t_range = (0, self.T, self.delta_t)

        self.t_axis = NumberLine(
            x_range=t_range,
            width=6,
            decimal_number_config={
                'font_size': 20,
                'num_decimal_places': 2,
            },
        )
        self.t_axis.add_numbers()
        # self.t_axis.to_edge(DOWN)

        self.price_axis = NumberLine(
            x_range=price_range,
            width=6,
            include_ticks=False,
            decimal_number_config={
                'font_size': 20,
                'num_decimal_places': 2,
            },
            line_to_number_direction=LEFT,
        )
        self.price_axis.rotate(90 * DEG, about_point=ORIGIN)

        prices = [
            self.S(i, j) for i in (self.n - 1, self.n) for j in range(i+1)
        ]
        prices.sort()

        ticks = VGroup()
        for price in prices:
            ticks.add(self.price_axis.get_tick(np.log(price), self.price_axis.tick_size))
        self.price_axis.add(ticks)

        for price, tick in zip(prices, ticks):
            label = DecimalNumber(price, font_size=20, num_decimal_places=2)
            always(label.next_to, tick, LEFT, SMALL_BUFF)
            self.price_axis.add(label)

        self.price_axis.to_edge(LEFT)

        self.t_axis.next_to(self.price_axis, RIGHT, LARGE_BUFF, BOTTOM)

        self.play(ShowCreation(self.t_axis))
        self.next_slide()
        self.play(ShowCreation(self.price_axis))
        self.next_slide()

        # write out formulas
        t2c = {
            r'\lambda': BLUE_A
        }
        u_formula = Tex(r'u = e^{\sigma \sqrt{\Delta t} + \lambda \sigma^2 \Delta t}', font_size=32, t2c=t2c)
        d_formula = Tex(r'd = e^{-\sigma \sqrt{\Delta t}  + \lambda \sigma^2 \Delta t}', font_size=32, t2c=t2c)

        lambda_label = Tex(r'\lambda', font_size=32)
        lambda_label.set_color(BLUE)

        formulas = VGroup(u_formula, d_formula).arrange(DOWN)
        formulas.next_to(self.c2p(0, self.y_max), RIGHT)
        # lambda_label.next_to(formulas.get_right(), RIGHT, MED_LARGE_BUFF)

        lambda_slider = NumberLine(
            x_range=(-5, 5, 1),
            include_numbers=True,
            width=5
        )
        lambda_slider.next_to(formulas, RIGHT, LARGE_BUFF)
        lambda_label.next_to(lambda_slider, DOWN)
        def get_tilt_on_slider():
            return lambda_slider.n2p(self.tilt.get_value())
        lambda_indicator = Dot(get_tilt_on_slider(), fill_color=BLUE)
        lambda_indicator.f_always.move_to(get_tilt_on_slider)
        
        lambda_value = DecimalNumber(self.tilt.get_value(), font_size=20, num_decimal_places=3)
        lambda_value.f_always.set_value(self.tilt.get_value)
        always(lambda_value.next_to, lambda_indicator, UP)

        self.play(Write(formulas))

        self.next_slide()

        self.play(
            Indicate(u_formula[r'\lambda \sigma^2 \Delta t']),
            Indicate(d_formula[r'\lambda \sigma^2 \Delta t']),
        )

        self.next_slide()

        self.play(ShowCreation(lambda_slider))
        self.play(GrowFromCenter(lambda_indicator), FadeIn(lambda_value), Write(lambda_label))

        self.next_slide()

        tree = self.constructTree(0)
        self.play(ShowCreation(tree))

        self.next_slide()

        reference_line = DashedLine(self.price_axis.n2p(np.log(self.asset_price)), self.c2p(self.T, self.asset_price))
        reference_line.set_stroke(color=ORANGE, width=2)
        reference_line.set_z_index(-5)

        strike_line = DashedLine(self.price_axis.n2p(np.log(self.strike_price)), self.c2p(self.T, self.strike_price))
        strike_line.set_stroke(color=RED, width=2)
        strike_line.set_z_index(-5)

        ref_label = Tex(r'S_0 = ' + str(self.asset_price), font_size=20)
        ref_label.set_color(ORANGE)
        strike_label = Tex(r'K = ' + str(self.strike_price), font_size=20)
        strike_label.set_color(ORANGE)
        ref_label.next_to(reference_line, RIGHT)
        strike_label.next_to(strike_line, RIGHT)

        self.play(ShowCreation(reference_line), Write(ref_label))

        self.next_slide()

        tree.add_updater(lambda mob: mob.become(self.constructTree(self.tilt.get_value())))
        self.play(self.tilt.animate.set_value(5), run_time=1.5)
        self.next_slide()
        self.play(self.tilt.animate.set_value(-5), run_time=3)
        self.next_slide()
        self.play(self.tilt.animate.set_value(0), run_time=1.5)

        self.next_slide()

        self.play(ShowCreation(strike_line), Write(strike_label))

        self.next_slide()

        eta = (np.log(self.strike_price / self.asset_price) - self.n * np.log(self.d0)) / np.log(self.u0 / self.d0)
        j0 = np.round(eta)
        strike_tilt = np.sqrt(self.delta_t) * 2 * (eta - j0) / self.asset_volatility / self.T

        self.play(self.tilt.animate.set_value(strike_tilt))

        self.next_slide()
